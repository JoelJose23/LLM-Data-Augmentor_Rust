use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::Path;
use tokio::sync::mpsc;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use rand::Rng;
use regex::Regex;

#[derive(Deserialize)]
struct PipelineRequest {
    pdf_path: String,
    augmentations: String, // Expecting "code" or "text"
}

#[derive(Serialize)]
struct PipelineResponse {
    status: String,
    message: String,
}

//Packet for the background workers
struct PipelineJob{
    pdf_path: String,
    augmentations: String,
}

#[derive(Clone)]
struct AppState {
    job_tx: mpsc::Sender<PipelineJob>,
}

#[tokio::main]
async fn main() {
    let (write_tx, mut write_rx) = mpsc::channel::<String>(100);
    let (job_tx, mut job_rx) = mpsc::channel::<PipelineJob>(100);

    tokio::spawn(async move {
        let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("Dataset.json")
        .await
        .expect("Failed to open or create dataset.json");

    println!("File Writer background task active. Monitoring Channels ...😊");

    while let Some(line) = write_rx.recv().await {
        if let Err(e) = file.write_all(line.as_bytes()).await {
            eprintln!("Failed to write to the file: {}", e);
        }
    } 
    });

    let worker_write_tx = write_tx.clone();

    tokio::spawn(async move{
        println!("Backround Process pipeline worker initialize. Waiting for the jobs...😊");
        while let Some(job) = job_rx.recv().await {
            println!("Background worker picked job for: {}", job.pdf_path);

            match extract_text_from_pdf(&job.pdf_path) {
                Ok(text) => {
                    let augment = apply_augmentations(&text, &job.augmentations).await;
                    let record = serde_json::json!({
                        "input_path": job.pdf_path,
                        "augmented_data": augment
                    });
                    let json_line = match serde_json::to_string_pretty(&record) {
                        Ok(json) => format!("{}\n", json),
                        Err(e) => {
                            eprintln!("Failed to serialize record: {}", e);
                            continue;
                        }
                    };
                    
                    if let Err(e) = worker_write_tx.send(json_line).await {
                        eprintln!("Failed to send augmented data to file writer: {}", e);
                    }
                }
                Err(err_msg) => {
                    println!("Error processing the request {}: {}", job.pdf_path, err_msg);
                }
                
            }
            
        }
    });

    let shared_state = AppState{ job_tx };

    let app = Router::new().route("/process", post(process_pdf_pipeline)).with_state(shared_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("The Backend is up boys 😎 at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn process_pdf_pipeline(State(state): State<AppState>, Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
    println!("Received request for: {}", payload.pdf_path);

    // 1. Extract the actual text from the PDF
    let job = PipelineJob{
        pdf_path: payload.pdf_path,
        augmentations:payload.augmentations
    };

    match state.job_tx.send(job).await {
        Ok(_) => {
            Json(PipelineResponse {
                status: "accepted".to_string(),
                message: "Job submitted successfully to background worker pipeline.".to_string(),
            })
        }
        Err(_) => Json(PipelineResponse {
            status: "error".to_string(),
            message: "Internal error: pipeline channel closed".to_string()
        }),
    }

}

fn extract_text_from_pdf(file_path: &str) -> Result<String, String> {
    if !Path::new(file_path).exists() {
        return Err(format!("File not found: {}", file_path));
    }

    // 1. Load the document safely
    let doc = lopdf::Document::load(file_path)
        .map_err(|e| format!("Failed to load PDF structure: {}", e))?;
    
    let mut full_text = String::new();
    
    // 2. Explicitly track and loop through every single page pointer
    let pages = doc.get_pages();
    let mut page_nums:Vec<u32>= pages.keys().copied().collect();
    page_nums.sort();
    for page_num in page_nums {
        // Extract text from just this single page
        match doc.extract_text(&[page_num]) {
            Ok(page_text) => {
                full_text.push_str(&page_text);
                full_text.push('\n'); // Maintain page boundaries
            }
            Err(e) => {
                // 💡 THE FIX: If page 4 fails, log it but KEEP GOING to page 5!
                eprintln!("Warning: Skipped messy layout on page {}: {}", page_num, e);
            }
        }
    }

    if full_text.is_empty() {
        return Err("No extractable text found in the entire PDF document.".to_string());
    }

    Ok(full_text)
}

// 3. The Augmentation Pipeline Engine
async fn apply_augmentations(text: &str, augmentation_type: &str) -> serde_json::Value {
    match augmentation_type.to_lowercase().as_str() {
        "text" => {
            let augmented = augment_conversational_text(text);
            let lines: Vec<&str> = augmented.lines().collect();
            serde_json::json!(lines) 
        }
        "code" => serde_json::json!(augment_code_syntax(text)),
        _ => serde_json::json!(text),
    }
}

// Augmentation Strategy 1: For Code Data (Variable Renaming Simulation)
fn augment_code_syntax(code: &str) -> String {
    // Regex looking for snake_case variables like "let user_id =" or "let mutable_var"
    let re = Regex::new(r"\b([a-z_][a-z0-9_]*)\b").unwrap();
    
    // Replace variable names globally to teach the model context over memory
    re.replace_all(code, |caps: &regex::Captures| {
        let word = &caps[1];
        if word == "let" || word == "fn" || word == "struct" || word == "match" {
            word.to_string() // Do not alter Rust keywords!
        } else {
            format!("{}_augmented", word) // Append suffix to randomize variables
        }
    }).into_owned()
}

// Augmentation Strategy 2: For Conversational Text (Synthetic Noise & Case Shifting)
fn augment_conversational_text(text: &str) -> String {
    let mut rng = rand::thread_rng();

    // Preserve original document order, but still occasionally vary casing.
    if rng.gen_bool(0.5) {
        text.to_lowercase()
    } else {
        text.to_string()
    }
}

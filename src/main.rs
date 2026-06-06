use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::Path;
use tokio::sync::mpsc;
use rand::Rng;
use rand::seq::SliceRandom;
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
    let (job_tx, mut job_rx) = mpsc::channel::<PipelineJob>(100);

    tokio::spawn(async move{
        println!("Backround Process pipeline worker initialize. Waiting for the jobs...😊");
        while let Some(job) = job_rx.recv().await {
            println!("Background worker picked job for: {}", job.pdf_path);

            match extract_text_from_pdf(&job.pdf_path) {
                Ok(text) => {
                    let augment = apply_augmentations(&text, &job.augmentations).await;
                    println!("Augmentation result for {}: {}", job.pdf_path, augment);
                }
                Err(err_msg) => {
                    println!("Error processing the request {}: {}", job.pdf_path, err_msg);
                }
                
            }
            
        }
    });

    let shared_state = Arc::new(AppState {job_tx: job_tx});

    let app = Router::new().route("/process", post(process_pdf_pipeline)).with_state(shared_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("The Backend is up boys 😎 at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn process_pdf_pipeline(State(state): State<Arc<AppState>>, Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
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
    match pdf_extract::extract_text(file_path) {
        Ok(text) => Ok(text),
        Err(e) => Err(format!("PDF Extraction failed: {}", e)),
    }
}

// 3. The Augmentation Pipeline Engine
async fn apply_augmentations(text: &str, augmentation_type: &str) -> String {
    match augmentation_type.to_lowercase().as_str() {
        "code" => augment_code_syntax(text),
        "text" => augment_conversational_text(text),
        _ => text.to_string(), // Fallback if no valid strategy is provided
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
    let mut lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    let mut rng = rand::thread_rng();

    // Strategy A: Shuffle sentences to make the pre-training model robust to document layout
    lines.shuffle(&mut rng);
    let shuffled_text = lines.join("\n");

    // Strategy B: Randomly lowercase blocks to simulate messy internet chat data
    if rng.gen_bool(0.5) {
        shuffled_text.to_lowercase()
    } else {
        shuffled_text
    }
}

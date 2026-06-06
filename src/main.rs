use axum::{
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::Path;
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
    extracted_text: String,
    augmentations_applied: String,
    augmentation_results: String,
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/process", post(process_pdf_pipeline));
    
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("The Backend is up boys 😎 at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn process_pdf_pipeline(Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
    println!("Received request for: {}", payload.pdf_path);

    // 1. Extract the actual text from the PDF
    match extract_text_from_pdf(&payload.pdf_path) {
        Ok(text) => {
            // 2. Pass the extracted text to the augmentation logic
            let augmented = apply_augmentations(&text, &payload.augmentations).await;
            
            Json(PipelineResponse {
                status: "success".to_string(),
                extracted_text: text,
                augmentations_applied: payload.augmentations,
                augmentation_results: augmented,
            })
        }
        Err(err_msg) => Json(PipelineResponse {
            status: "error".to_string(),
            extracted_text: "".to_string(),
            augmentations_applied: payload.augmentations,
            augmentation_results: err_msg,
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

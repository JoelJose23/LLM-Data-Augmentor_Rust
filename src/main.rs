use axum::{
    Json, Router, routing::{get, post}
};
use eframe::egui::debug_text::print;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

//1. Define the expected structure of the data
#[derive(Deserialize)]
struct PipelineRequest {
    pdf_path: String,
    augmentations: String,
}

#[derive(Serialize)]
struct PipelineResponse {
    status: String,
    extracted_text: String,
    augmentations_applied: String,
    augmentation_results: String
}

//2. Define the POST handler
#[tokio::main]
async fn main(){
    let app = Router::new()
        .route("/process", post(process_pdf_pipeline));
    
    // Run it on local port 3000
    let addr = SocketAddr::from (([127, 0, 0, 1], 3000));
    println!("The Backend is up boys 😎 at https://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

//3. Define the helper function process_pdf_pipeline
async fn process_pdf_pipeline(Json(payload): Json<PipelineRequest>) -> Json<PipelineResponse> {
    println!("Received the Request to process at:{}", payload.pdf_path);
    let response = PipelineResponse {
        status: "success".to_string(),
        extracted_text: format!("Mock text from {}", payload.pdf_path),
        augmentations_applied: payload.augmentations.clone(),
        // Added `{:?}` here to format the Vector safely:
        augmentation_results: format!("Mock results for augmentations: {:?}", payload.augmentations)
    };
    Json(response)
}

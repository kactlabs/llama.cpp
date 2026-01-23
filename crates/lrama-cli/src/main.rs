//! lrama-cli: Rust-based LLaMA inference CLI
//!
//! A pure Rust implementation of LLaMA model inference using GGUF format.

use anyhow::{Context, Result};
use clap::Parser;
use gguf::GGUFReader;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "lrama-cli")]
#[command(about = "Rust-based LLaMA inference CLI", long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text
    #[arg(short, long)]
    prompt: Option<String>,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value = "128")]
    n_predict: usize,

    /// Temperature for sampling
    #[arg(short, long, default_value = "0.8")]
    temperature: f32,

    /// Show model information only
    #[arg(long)]
    info: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🦙 lrama-cli - Rust LLaMA Inference");
    println!("=====================================\n");

    // Load GGUF model
    println!("Loading model: {}", args.model.display());
    let reader = GGUFReader::open(&args.model)
        .context("Failed to open GGUF file")?;

    // Display model information
    print_model_info(&reader)?;

    if args.info {
        return Ok(());
    }

    // Get prompt
    let prompt = args.prompt.unwrap_or_else(|| {
        "Once upon a time".to_string()
    });

    println!("\n📝 Prompt: {}", prompt);
    println!("🎲 Temperature: {}", args.temperature);
    println!("🔢 Generating {} tokens\n", args.n_predict);

    // TODO: Implement actual inference
    println!("⚠️  Inference not yet implemented!");
    println!("   Next steps:");
    println!("   1. Tokenize prompt");
    println!("   2. Load model weights into tensors");
    println!("   3. Run forward pass");
    println!("   4. Sample next token");
    println!("   5. Repeat for n_predict tokens");

    Ok(())
}

fn print_model_info(reader: &GGUFReader) -> Result<()> {
    println!("📊 Model Information:");
    println!("   Version: {:?}", reader.version());
    println!("   Tensors: {}", reader.tensor_count());

    // Print key metadata
    if let Some(name) = reader.get_metadata("general.name") {
        println!("   Name: {:?}", name);
    }
    if let Some(arch) = reader.get_metadata("general.architecture") {
        println!("   Architecture: {:?}", arch);
    }

    // Print some tensor names
    let tensor_names = reader.tensor_names();
    println!("\n📦 Sample tensors ({} total):", tensor_names.len());
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        if let Some(info) = reader.get_tensor_info(name) {
            println!("   {}. {} - {:?} {:?}", 
                     i + 1, name, info.tensor_type, info.dimensions);
        }
    }
    if tensor_names.len() > 10 {
        println!("   ... and {} more", tensor_names.len() - 10);
    }

    Ok(())
}

use anyhow::Result;
use clap::Parser;
use gguf::GGUFReader;

#[derive(Parser)]
#[command(name = "gguf-info")]
#[command(about = "Display information about a GGUF model file")]
struct Args {
    /// Path to the GGUF file
    #[arg(value_name = "FILE")]
    file: String,

    /// Show detailed tensor information
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading GGUF file: {}", args.file);
    println!();

    let reader = GGUFReader::open(&args.file)?;

    // Print version info
    println!("GGUF Version: {:?}", reader.version());
    println!("Tensor Count: {}", reader.tensor_count());
    println!();

    // Print metadata
    println!("=== Metadata ===");
    let metadata = reader.metadata();
    
    // Print important metadata first
    let important_keys = [
        "general.name",
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
        "general.parameter_count",
    ];

    for key in &important_keys {
        if let Some(value) = metadata.get(*key) {
            print_metadata_value(key, value);
        }
    }

    println!();
    println!("=== All Metadata ({} keys) ===", metadata.len());
    let mut keys: Vec<_> = metadata.keys().collect();
    keys.sort();
    
    for key in keys {
        if !important_keys.contains(&key.as_str()) {
            print_metadata_value(key, metadata.get(key).unwrap());
        }
    }

    println!();
    println!("=== Tensors ===");
    
    let tensor_names = reader.tensor_names();
    println!("Total tensors: {}", tensor_names.len());
    
    if args.verbose {
        println!();
        for name in tensor_names {
            if let Some(info) = reader.get_tensor_info(name) {
                println!("  {}", name);
                println!("    Type: {:?}", info.tensor_type);
                println!("    Dimensions: {:?}", info.dimensions);
                println!("    Elements: {}", info.n_elements());
                println!("    Size: {} bytes", info.size_bytes());
                println!();
            }
        }
    } else {
        // Just show first few
        for name in tensor_names.iter().take(5) {
            if let Some(info) = reader.get_tensor_info(name) {
                println!("  {} - {:?} {:?}", name, info.tensor_type, info.dimensions);
            }
        }
        if tensor_names.len() > 5 {
            println!("  ... and {} more (use --verbose to see all)", tensor_names.len() - 5);
        }
    }

    Ok(())
}

fn print_metadata_value(key: &str, value: &gguf::MetadataValue) {
    use gguf::MetadataValue;
    
    match value {
        MetadataValue::UInt8(v) => println!("  {}: {} (u8)", key, v),
        MetadataValue::Int8(v) => println!("  {}: {} (i8)", key, v),
        MetadataValue::UInt16(v) => println!("  {}: {} (u16)", key, v),
        MetadataValue::Int16(v) => println!("  {}: {} (i16)", key, v),
        MetadataValue::UInt32(v) => println!("  {}: {} (u32)", key, v),
        MetadataValue::Int32(v) => println!("  {}: {} (i32)", key, v),
        MetadataValue::UInt64(v) => println!("  {}: {} (u64)", key, v),
        MetadataValue::Int64(v) => println!("  {}: {} (i64)", key, v),
        MetadataValue::Float32(v) => println!("  {}: {} (f32)", key, v),
        MetadataValue::Float64(v) => println!("  {}: {} (f64)", key, v),
        MetadataValue::Bool(v) => println!("  {}: {} (bool)", key, v),
        MetadataValue::String(v) => println!("  {}: \"{}\"", key, v),
        MetadataValue::Array(arr) => {
            println!("  {}: [{:?} array, {} elements]", key, arr.value_type, arr.values.len());
        }
    }
}

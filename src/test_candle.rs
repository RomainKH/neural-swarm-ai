use candle_transformers::models::quantized_llama::ModelWeights;
fn inspect(m: &mut ModelWeights) {
    let _ = m.forward_layer();
}

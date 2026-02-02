class AudioProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        // inputs is an array of inputs, each input is an array of channels
        // We expect one input, and we'll just take the first channel (mono)
        const input = inputs[0];
        if (input && input.length > 0) {
            const channelData = input[0]; // Float32Array
            // Post the data to the main thread
            // We slice it to ensure we aren't sending a detached buffer if the browser reuses it, 
            // though usually port.postMessage copies.
            if (channelData.length > 0) {
                this.port.postMessage(channelData);
            }
        }
        return true; // Keep the processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);

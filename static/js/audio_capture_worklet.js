// AudioWorklet replacement for the legacy ScriptProcessorNode-based capture
// in host.js and viewer.js. Posts each 128-sample mono Float32 frame back to
// the main thread, which downsamples and forwards as PCM16 over the
// WebSocket. The processor itself stays minimal so it can run in the audio
// thread without missing deadlines.
//
// Buffering note: AudioWorklet hands us 128 frames at a time at the
// AudioContext rate (e.g. 48 kHz → ~2.7 ms). Sending one postMessage per
// frame is fine in practice but the main-thread side accumulates samples
// before downsample/encode, so end-to-end latency is unchanged from the
// 4096-sample ScriptProcessor it replaces.

class AudioCaptureProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        const channel = input?.[0];
        if (channel && channel.length > 0) {
            // Float32Array is transferable; clone so the AudioContext can
            // safely reuse its internal buffer for the next render quantum.
            this.port.postMessage(channel.slice());
        }
        return true;
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);

// PCM resampling + 16-bit conversion helpers shared between host.js (the
// ScriptProcessor mic path) and viewer.js (the AudioWorklet mic path).
// Both pages downsample the browser's native rate (typically 48 kHz) to
// the 16 kHz stream the server expects, then pack the float samples into
// signed 16-bit little-endian PCM for ws.send().

(() => {
    /**
     * Resample a Float32 audio buffer from one sample rate to another.
     * Uses a stride/blend strategy: pick samples at fractional indices
     * and average each block to mitigate aliasing. Sufficient for speech
     * (no reconstruction filter needed for the 48k→16k case we hit).
     *
     * @param {Float32Array} buffer  Source samples in [-1, 1].
     * @param {number} srcRate       Source sample rate (Hz).
     * @param {number} dstRate       Target sample rate (Hz).
     * @returns {Float32Array}
     */
    function downsampleBuffer(buffer, srcRate, dstRate) {
        if (dstRate === srcRate) return buffer;
        if (dstRate > srcRate) {
            throw new Error(`downsampleBuffer cannot upsample (${srcRate} -> ${dstRate})`);
        }
        const ratio = srcRate / dstRate;
        const newLength = Math.floor(buffer.length / ratio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < newLength) {
            const nextOffsetBuffer = Math.floor((offsetResult + 1) * ratio);
            let accum = 0;
            let count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count += 1;
            }
            result[offsetResult] = count > 0 ? accum / count : 0;
            offsetResult += 1;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }

    /**
     * Pack a Float32 sample buffer (range [-1, 1]) into signed 16-bit
     * little-endian PCM, returning the underlying ArrayBuffer so callers
     * can hand it straight to WebSocket.send().
     *
     * @param {Float32Array} input
     * @returns {ArrayBuffer}
     */
    function floatTo16BitPCM(input) {
        const output = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return output.buffer;
    }

    /** @type {any} */ (window).LinguaGapAudio = { downsampleBuffer, floatTo16BitPCM };
})();

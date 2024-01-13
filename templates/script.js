function transcribeAndDiarize() {
    const fileInput = document.getElementById("fileInput");
    const transcriptionResult = document.getElementById("transcriptionResult");
    const result = document.getElementById("result");

    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a video file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/transcribe_and_diarize", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        transcriptionResult.style.display = "block";
        result.innerText = data;
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred during transcription and diarization.");
    });
}

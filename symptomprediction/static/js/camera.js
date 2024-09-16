let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let context = canvas.getContext("2d");
let capturedImage = document.getElementById("capturedImage");
let captureButton = document.getElementById("captureButton");
let miniCanvas = document.getElementById("miniCanvas");
let miniContext = miniCanvas.getContext("2d");

captureButton.style.display = "none";

function startCamera() {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(stream => {
        video.srcObject = stream;
        video.play();
        captureButton.style.display = "block";
    }).catch(err => {
        console.error('Error accessing camera: ', err);
    });
}

function captureImage() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    capturedImage.src = canvas.toDataURL("image/png");
    capturedImage.style.display = "block";
    canvas.style.display = "none";
    video.style.display = "none";
    stopCamera();
}

function stopCamera() {
    if (video.srcObject) {
        let stream = video.srcObject;
        let tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        captureButton.style.display = "none";
    }
}

function drawMiniFrame() {
    miniContext.drawImage(video, 0, 0, miniCanvas.width, miniCanvas.height);
    requestAnimationFrame(drawMiniFrame);
}

drawMiniFrame();
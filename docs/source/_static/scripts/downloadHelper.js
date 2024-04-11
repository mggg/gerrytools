document.addEventListener('DOMContentLoaded', function() { 
    document.querySelectorAll('.download-badge').forEach(button => {
        button.addEventListener('click', function() {
            const url = this.getAttribute('data-url');
            const filename = this.getAttribute('data-filename');
            initiateFileDownload(url, filename);
        });
    });
});

function initiateFileDownload(url, filename) {
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const blobUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(blobUrl);
        })
        .catch(console.error); // Handle any errors
}


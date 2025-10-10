<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Topic Modeling</title>
</head>
<body>
<h1>Topic Modeling</h1>
<form action="index.php" method="POST" enctype="multipart/form-data">
    <label for="username">Keywords:</label>
    <input type="file" name="keywords" accept=".csv"><br>
    <label for="username">Comments:</label>
    <input type="file" name="comments" accept=".csv"><br>
    <button type="submit">Upload</button>
</form>
</body>
</html>

<?php
function process_file($file,$file_name){
    // Check for upload errors
    if ($file['error'] !== UPLOAD_ERR_OK) {
        echo "File upload error: " . $file['error'];
        exit;
    }

    // Debug paths
    $uploadDir = 'uploads/';
    //$filePath = $uploadDir . basename($file['name']);
    $filePath = $uploadDir . $file_name;

    // Ensure upload directory exists and is writable
    if (!is_dir($uploadDir)) {
        mkdir($uploadDir, 0777, true);
    }
    
    if (!is_writable($uploadDir)) {
        echo "Upload directory is not writable.";
        exit;
    }

    // Move the uploaded file
    if (move_uploaded_file($file['tmp_name'], $filePath)) {
        echo "File uploaded successfully: $filePath";
    } else {
        echo "Failed to move uploaded file.";
    }
    
}
// Check if a file was submitted
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['keywords'])) {
    $file = $_FILES['keywords'];
    process_file($file,"keywords.csv");
} else {
    echo "No file uploaded.";
}
?>
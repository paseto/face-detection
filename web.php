<?php
ini_set('display_errors', 1);

$img = $_FILES['picture']['tmp_name'];

//move_uploaded_file($img, "pic.png");

try{
    $command = escapeshellcmd('python3 /home/ubuntu/face-detection/detect.py '.$img);
    $raw = shell_exec($command);
    error_log($raw, 3, 'log.log');
    echo base64_encode(file_get_contents($raw))
}catch(Exception $e){
    echo $e->getMessage();
}



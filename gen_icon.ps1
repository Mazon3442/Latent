Add-Type -AssemblyName System.Drawing
$img = New-Object System.Drawing.Bitmap(16, 16)
$g = [System.Drawing.Graphics]::FromImage($img)
$g.Clear([System.Drawing.ColorTranslator.FromHtml('#111213')])
$img.Save('C:\Users\mason\Documents\Zed\Latent\app\src-tauri\icons\icon.ico', [System.Drawing.Imaging.ImageFormat]::Icon)
$img.Dispose()
$g.Dispose()
Write-Host 'Created icon.ico'

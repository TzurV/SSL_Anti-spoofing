
REM docker run -v "C:\work\Github\Tensor_Lihu\MultiKol\SSL_Anti-spoofing\model:/app/model" ^
REM            –v "C:\work\Github\Tensor_Lihu\MultiKol\ASVspoof2021\ASVspoof2021_LA_eval:/app/asvspoof2021_la_eval" ^
REM            –v "C:\work\Github\Tensor_Lihu\MultiKol\SSL_Anti-spoofing\results:/app/results" ^
REM            -it build1
           
docker run -v "C:\work\Github\Tensor_Lihu\MultiKol\SSL_Anti-spoofing\results:/app/results" ^
           -v "C:\work\Github\Tensor_Lihu\MultiKol\SSL_Anti-spoofing\model:/app/model" ^
           -v "C:\work\Github\Tensor_Lihu\MultiKol\ASVspoof2021\ASVspoof2021_LA_eval:/app/ASVspoof2021_LA_eval" ^
           -v "C:\work\Github\Tensor_Lihu\MultiKol\ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval:/app/ASVspoof2021_DF_eval" ^
           -it build1

REM            -v "C:\work\Github\Tensor_Lihu\MultiKol\ASVspoof2021\ASVspoof2021_DF_eval_part01:/app/ASVspoof2021_DF_eval_part01"
           

### install tesseract
- visit https://github.com/UB-Mannheim/tesseract/wiki
- download `tesseract-ocr-w64-setup-v5.0.0-alpha.20201127.exe`
- run `.exe` file
- "Additional script data" check `hangul script`, `hangul vertical script`
- "Additional language data" check `Korean`
- 환경변수 경로 추가
  - Tesseract 설치 경로 환경변수에 추가 (default: `C:\Program Files\Tesseract-OCR`
  - 프롬프트 명령어로 설치 확인

```
$ tesseract --version
tesseract v5.0.0-alpha.20201127
...
``` 

### requirements
- dlib's weight file
  - visit http://dlib.net/files/
  - download `shape_predictor_68_face_landmarks.dat.bz2`
  - upzip `shape_predictor_68_face_landmarks.dat.bz2` -> `shape_predictor_68_face_landmarks.dat`
  - place weight file into `~/weight/`

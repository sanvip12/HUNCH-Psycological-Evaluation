

\# Convert Youtube Video to audio  
\!pip install yt-dlp==2023.7.6

import yt\_dlp

def download\_youtube\_audio(url, output\_name='downloaded\_song.mp3'):  
    """  
    Download YouTube video and convert to audio  
    """  
    ydl\_opts \= {  
        'format': 'bestaudio/best',  
        'postprocessors': \[{  
            'key': 'FFmpegExtractAudio',  
            'preferredcodec': 'mp3',  
            'preferredquality': '192',  
        }\],  
        'outtmpl': output\_name.replace('.mp3', ''),  
        'quiet': False,  
        'verbose': True,  \# Add verbose output for debugging  
    }  
      
    try:  
        print(f"Attempting to download {url}...")  
        with yt\_dlp.YoutubeDL(ydl\_opts) as ydl:  
            ydl.download(\[url\])  
        print(f"\\  
Successfully downloaded to {output\_name}")  
        return True  
    except Exception as e:  
        print(f"\\  
Error occurred: {str(e)}")  
        print("\\  
Trying alternative method...")  
        try:  
            \# Try alternative options  
            ydl\_opts\['format'\] \= 'bestaudio'  
            with yt\_dlp.YoutubeDL(ydl\_opts) as ydl:  
                ydl.download(\[url\])  
            print(f"\\  
Successfully downloaded to {output\_name}")  
            return True  
        except Exception as e2:  
            print(f"\\  
Alternative method also failed: {str(e2)}")  
            return False

\# Try downloading  
url \= "https://youtu.be/EIdjBnX0vec"  
download\_youtube\_audio(url, "your\_song.mp3")  

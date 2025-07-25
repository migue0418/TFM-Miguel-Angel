// components/AudioRecorder.js
import React, { useRef, useState } from "react";

const AudioRecorder = ({ onRecordingComplete }) => {
  const mediaRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new window.MediaRecorder(stream);
    const chunks = [];
    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setAudioURL(url);
      onRecordingComplete && onRecordingComplete(blob);
    };
    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current && mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  return (
    <div>
      {!isRecording ? (
        <button type="button" onClick={startRecording}>Grabar audio</button>
      ) : (
        <button type="button" onClick={stopRecording}>Detener</button>
      )}
      {audioURL && (
        <audio src={audioURL} controls />
      )}
    </div>
  );
};

export default AudioRecorder;

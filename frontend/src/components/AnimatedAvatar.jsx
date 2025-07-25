import React, { useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import api from "../functions/api";

const BLENDSHAPE_MAP = {
  "blendShapes.EyeBlinkLeft":      "A14_Eye_Blink_Left",
  "blendShapes.EyeBlinkRight":     "A15_Eye_Blink_Right",
  "blendShapes.EyeSquintLeft":     "A16_Eye_Squint_Left",
  "blendShapes.EyeSquintRight":    "A17_Eye_Squint_Right",
  "blendShapes.EyeWideLeft":       "A18_Eye_Wide_Left",
  "blendShapes.EyeWideRight":      "A19_Eye_Wide_Right",
  "blendShapes.JawOpen":           "A25_Jaw_Open",
  "blendShapes.JawLeft":           "A27_Jaw_Left",
  "blendShapes.JawRight":          "A28_Jaw_Right",
  "blendShapes.JawForward":        "A26_Jaw_Forward",
  "blendShapes.MouthSmileLeft":    "A38_Mouth_Smile_Left",
  "blendShapes.MouthSmileRight":   "A39_Mouth_Smile_Right",
  "blendShapes.MouthFrownLeft":    "A40_Mouth_Frown_Left",
  "blendShapes.MouthFrownRight":   "A41_Mouth_Frown_Right",
  "blendShapes.MouthDimpleLeft":   "A42_Mouth_Dimple_Left",
  "blendShapes.MouthDimpleRight":  "A43_Mouth_Dimple_Right",
  "blendShapes.MouthStretchLeft":  "A50_Mouth_Stretch_Left",
  "blendShapes.MouthStretchRight": "A51_Mouth_Stretch_Right",
  "blendShapes.MouthPucker":       "A30_Mouth_Pucker",
  "blendShapes.MouthFunnel":       "A29_Mouth_Funnel",
  "blendShapes.MouthLeft":         "A31_Mouth_Left",
  "blendShapes.MouthRight":        "A32_Mouth_Right",
  "blendShapes.MouthClose":        "A37_Mouth_Close",
  "blendShapes.MouthPressLeft":    "A48_Mouth_Press_Left",
  "blendShapes.MouthPressRight":   "A49_Mouth_Press_Right",
  "blendShapes.MouthRollLower":    "A34_Mouth_Roll_Lower",
  "blendShapes.MouthRollUpper":    "A33_Mouth_Roll_Upper",
  "blendShapes.MouthShrugLower":   "A36_Mouth_Shrug_Lower",
  "blendShapes.MouthShrugUpper":   "A35_Mouth_Shrug_Upper",
  "blendShapes.MouthLowerDownLeft": "A46_Mouth_Lower_Down_Left",
  "blendShapes.MouthLowerDownRight": "A47_Mouth_Lower_Down_Right",
  "blendShapes.MouthUpperUpLeft":   "A44_Mouth_Upper_Up_Left",
  "blendShapes.MouthUpperUpRight":  "A45_Mouth_Upper_Up_Right",
  "blendShapes.BrowDownLeft":      "A02_Brow_Down_Left",
  "blendShapes.BrowDownRight":     "A03_Brow_Down_Right",
  "blendShapes.BrowInnerUp":       "A01_Brow_Inner_Up",
  "blendShapes.BrowOuterUpLeft":   "A04_Brow_Outer_Up_Left",
  "blendShapes.BrowOuterUpRight":  "A05_Brow_Outer_Up_Right",
  "blendShapes.CheekSquintLeft":   "A21_Cheek_Squint_Left",
  "blendShapes.CheekSquintRight":  "A22_Cheek_Squint_Right",
  "blendShapes.NoseSneerLeft":     "A23_Nose_Sneer_Left",
  "blendShapes.NoseSneerRight":    "A24_Nose_Sneer_Right",
  "blendShapes.CheekPuff":         "A20_Cheek_Puff",
  "blendShapes.TongueOut":         "A52_Tongue_Out",
  "blendShapes.TongueTipUp": "T06_Tongue_Tip_Up",
  "blendShapes.TongueTipDown": "T07_Tongue_Tip_Down",
  "blendShapes.TongueUp": "T01_Tongue_Up",
  "blendShapes.TongueDown": "T02_Tongue_Down",
  "blendShapes.TongueLeft": "T03_Tongue_Left",
  "blendShapes.TongueRight": "T04_Tongue_Right",
  "blendShapes.TongueWide": "T08_Tongue_Width",
  "blendShapes.TongueNarrow": "V_Tongue_Narrow"
};

// --- Utils ---
function base64ToBlob(base64, type = "audio/wav") {
  const binary = atob(base64);
  const array = [];
  for (let i = 0; i < binary.length; i++) array.push(binary.charCodeAt(i));
  return new Blob([new Uint8Array(array)], { type });
}

function extractBlendshapeNames(frame) {
  return Object.keys(frame).filter((k) => k.startsWith("blendShapes."));
}
function framesToArray(frames, morphTargets) {
  return frames.map((frame) => morphTargets.map((name) => frame[name] || 0));
}
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// --- Modelo animado GLB ---
function AvatarGLBModel({ blendshapes, audioRef, modelUrl }) {
  const { scene } = useGLTF(modelUrl);
  const morphTargets = extractBlendshapeNames(blendshapes[0]);
  const framesArr = framesToArray(blendshapes, morphTargets);
  const frameTime = 1 / 30;

  useFrame(() => {
    if (!audioRef.current || !framesArr.length) return;
    const audio = audioRef.current;
    const frameIdx = clamp(Math.floor(audio.currentTime / frameTime), 0, framesArr.length - 1);
    scene.traverse((obj) => {
      if (obj.isMesh && obj.morphTargetDictionary) {
        morphTargets.forEach((name, i) => {
          const targetName = BLENDSHAPE_MAP[name] || name.replace("blendShapes.", "");
          const idx = obj.morphTargetDictionary[targetName];
          if (idx !== undefined) obj.morphTargetInfluences[idx] = framesArr[frameIdx][i];
        });
      }
    });
  });

  // Rotar, escalar y centrar el modelo solo una vez
  useEffect(() => {
    scene.rotation.x = -Math.PI / 2; // Ajusta si necesario
    scene.scale.set(1, 1, 1); // Usa 1 si tu GLB ya está a escala real
    scene.position.y = 0.0;
  }, [scene]);

  return <primitive object={scene} />;
}

// --- Componente principal ---
const AnimatedAvatar = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [blendshapes, setBlendshapes] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef();

  // Usa la ruta correcta a tu GLB (asegúrate de que está en /public/models)
  const MODEL_GLB_URL = "/app/models/boy.glb";

  const handleAudioUpload = (e) => {
    if (e.target.files[0]) setAudioFile(e.target.files[0]);
  };

  const handleGenerateAnimation = async () => {
    if (!audioFile) return alert("Sube un archivo de audio primero.");
    const formData = new FormData();
    formData.append("audio_file", audioFile);

    const token = localStorage.getItem("access_token");
    const response = await api.post(
      "/nvidia/avatar/blendshapes-audio",
      formData,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );

    setBlendshapes(response.data.blendshapes);
    const blob = base64ToBlob(
      response.data.audio_base64,
      response.data.audio_type || "audio/wav"
    );
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
  };

  return (
    <section style={{ maxWidth: 900, margin: "0 auto" }}>
      <h2>Avatar 3D Animado (Audio2Face)</h2>
      <div style={{ marginBottom: 16 }}>
        <input type="file" accept="audio/*" onChange={handleAudioUpload} />
        <button onClick={handleGenerateAnimation} disabled={!audioFile}>
          Generar animación
        </button>
      </div>
      <div style={{ height: 600, background: "#181818" }}>
        {blendshapes && audioUrl ? (
          <Canvas camera={{ position: [0, 1.65, 0.7], fov: 28 }}>
            <ambientLight intensity={1.0} />
            <directionalLight position={[2, 10, 2]} intensity={2} />
            <AvatarGLBModel
              blendshapes={blendshapes}
              audioRef={audioRef}
              modelUrl={MODEL_GLB_URL}
            />
            <OrbitControls target={[0, 1.65, 0]} minDistance={0.5} maxDistance={1.5}/>
          </Canvas>
        ) : (
          <div style={{ color: "#fff", padding: 32 }}>
            Sube un audio y pulsa “Generar animación”.
          </div>
        )}
      </div>
      {audioUrl && (
        <audio
          ref={audioRef}
          controls
          src={audioUrl}
          style={{ marginTop: 16 }}
        />
      )}
    </section>
  );
};

export default AnimatedAvatar;

import { useState, useRef, useEffect } from "react"
import axios from "axios"
import "./App.css"
import Stems8View from "./Stems8"

const API = "http://localhost:8000/api"

// ── Spatial Map Component ─────────────────────────────────────────────────────
function SpatialMap({ instruments, currentTime }) {
  const size = 300
  const cx   = size / 2
  const cy   = size / 2
  const r    = 110

  const toXY = (azimuth, distance) => {
    const rad = ((azimuth - 90) * Math.PI) / 180
    const d   = r * (0.4 + distance * 0.6)
    return { x: cx + d * Math.cos(rad), y: cy + d * Math.sin(rad) }
  }

  const COLORS = {
    "Plucked strings": "#F59E0B",
    "Guitar":          "#F59E0B",
    "Acoustic guitar": "#F59E0B",
    "Guitar strum":    "#F59E0B",
    "Vocals":          "#60A5FA",
    "Singing":         "#60A5FA",
    "Music":           "#94A3B8",
    "Instrument":      "#94A3B8",
    "Cinematic":       "#A78BFA",
    "Swing music":     "#A78BFA",
    "Strings":         "#34D399",
    "Tabla":           "#F87171",
    "Percussion":      "#F87171",
    "Flute":           "#86EFAC",
    "Piano":           "#FCA5A5",
  }

  return (
    <div className="spatial-map-wrap">
      <svg width={size} height={size}>
        {/* Rings */}
        {[0.33, 0.66, 1].map((s, i) => (
          <circle key={i} cx={cx} cy={cy} r={r * s}
            fill="none" stroke="#1e293b" strokeWidth={1} />
        ))}
        {/* Direction labels */}
        {[["N", cx, cy - r - 14], ["S", cx, cy + r + 18],
          ["E", cx + r + 14, cy + 5], ["W", cx - r - 14, cy + 5]].map(([l, x, y]) => (
          <text key={l} x={x} y={y} textAnchor="middle"
            fill="#475569" fontSize={11}>{l}</text>
        ))}
        {/* Listener head */}
        <circle cx={cx} cy={cy} r={14} fill="#0f172a" stroke="#334155" strokeWidth={2} />
        <text x={cx} y={cy + 5} textAnchor="middle" fill="#94a3b8" fontSize={11}>👤</text>
        {/* Instruments */}
        {instruments.map((instr, i) => {
          const { x, y } = toXY(instr.azimuth, instr.distance)
          const color     = COLORS[instr.label] || "#64748b"
          const isMoving  = instr.in_motion
          return (
            <g key={i}>
              {isMoving && (
                <circle cx={x} cy={y} r={16} fill="none"
                  stroke={color} strokeWidth={1.5} opacity={0.4}
                  style={{ animation: "pulse 1s infinite" }} />
              )}
              <circle cx={x} cy={y} r={10} fill={color}
                opacity={0.9} stroke="#0f172a" strokeWidth={1.5} />
              <text x={x} y={y + 22} textAnchor="middle"
                fill={color} fontSize={9} fontWeight={isMoving ? "bold" : "normal"}>
                {instr.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ── Timeline Bar Component ────────────────────────────────────────────────────
function TimelineBar({ motionEvents, duration, currentTime }) {
  if (!motionEvents || duration === 0) return null
  return (
    <div className="timeline-wrap">
      <div className="timeline-label">Motion timeline</div>
      <div className="timeline-bar">
        {motionEvents.map((ev, i) => {
          const left  = (ev.start_sec / duration) * 100
          const width = ((ev.end_sec - ev.start_sec) / duration) * 100
          const color = ev.spatial_motion === "circular" ? "#F59E0B"
                      : ev.spatial_motion === "right_to_left" ? "#60A5FA" : "#34D399"
          return (
            <div key={i} className="timeline-event" style={{
              left: left + "%", width: width + "%", background: color,
            }} title={ev.dominant_instrument + " — " + ev.spatial_motion} />
          )
        })}
        {/* Playhead */}
        <div className="timeline-playhead" style={{
          left: (currentTime / duration) * 100 + "%"
        }} />
      </div>
      <div className="timeline-legend">
        {motionEvents.some((ev) => ev.spatial_motion === "original_pan")
          ? <span style={{color:"#34D399"}}>● movement measured in the original stereo</span>
          : motionEvents.length === 0
            ? <span>No movement (switched off, or none found in the original)</span>
            : <>
                <span style={{color:"#F59E0B"}}>● circular</span>
                <span style={{color:"#60A5FA"}}>● right→left</span>
                <span style={{color:"#34D399"}}>● left→right</span>
              </>}
      </div>
    </div>
  )
}

// ── Detection Summary Component ───────────────────────────────────────────────
function DetectionPanel({ summary }) {
  if (!summary || summary.length === 0) return null
  const max = summary[0].avg_confidence
  return (
    <div className="detection-panel">
      <div className="panel-title">Detected instruments</div>
      {summary.slice(0, 8).map((item, i) => (
        <div key={i} className="det-row">
          <div className="det-name">{item.label}</div>
          <div className="det-bar-wrap">
            <div className="det-bar" style={{
              width: (item.avg_confidence / max * 100) + "%"
            }} />
          </div>
          <div className="det-pct">{Math.round(item.avg_confidence * 100)}%</div>
          <div className="det-wins">{item.window_count}w</div>
        </div>
      ))}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [file,         setFile]         = useState(null)
  const [jobId,        setJobId]        = useState(null)
  const [status,       setStatus]       = useState("idle")
  const [progress,     setProgress]     = useState(0)
  const [message,      setMessage]      = useState("")
  const [result,       setResult]       = useState(null)
  const [currentTime,  setCurrentTime]  = useState(0)
  const [instruments,  setInstruments]  = useState([])
  const [dragging,     setDragging]     = useState(false)
  const [motionOn,     setMotionOn]     = useState(false)
  const audioRef  = useRef(null)
  const pollRef   = useRef(null)
  const timelineRef = useRef(null)

  // ── File drop / select ────────────────────────────────────────────────────
  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  const handleFileInput = (e) => {
    if (e.target.files[0]) setFile(e.target.files[0])
  }

  // ── Upload and start pipeline ─────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return
    const form = new FormData()
    form.append("file", file)
    form.append("motion", motionOn ? "true" : "false")
    setStatus("uploading"); setProgress(5); setMessage("Uploading...")
    try {
      const res = await axios.post(API + "/upload", form)
      setJobId(res.data.job_id)
      setStatus("running"); setProgress(10)
      startPolling(res.data.job_id)
    } catch (err) {
      setStatus("error"); setMessage("Upload failed: " + err.message)
    }
  }

  // ── Poll status ───────────────────────────────────────────────────────────
  const startPolling = (jid) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(API + "/status/" + jid)
        const pct = res.data.progress
        setProgress(pct)
        setMessage(res.data.message)
        if (res.data.status === "complete") {
          clearInterval(pollRef.current)
          setStatus("complete"); setProgress(100)
          fetchResult(jid)
        } else if (res.data.status === "error") {
          clearInterval(pollRef.current)
          setStatus("error"); setMessage(res.data.message)
        }
      } catch (err) {
        clearInterval(pollRef.current)
        setStatus("error"); setMessage("Polling failed")
      }
    }, 2000)
  }

  // ── Reopen a finished job from the address bar (?job=<id>) ────────────────
  useEffect(() => {
    const jid = new URLSearchParams(window.location.search).get("job")
    if (jid) {
      setJobId(jid); setStatus("complete"); setProgress(100)
      fetchResult(jid)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Fetch full result ─────────────────────────────────────────────────────
  const fetchResult = async (jid) => {
    try {
      const res = await axios.get(API + "/result/" + jid)
      setResult(res.data)
      // keep the job in the address bar so a page reload shows it again
      window.history.replaceState(null, "", "?job=" + jid)
      if (res.data.spatial_timeline && res.data.spatial_timeline.length > 0) {
        setInstruments(res.data.spatial_timeline[0].instruments || [])
      }
    } catch (err) {
      setStatus("error")
      setMessage("Could not load result: " + err.message)
    }
  }

  // ── Audio time sync with spatial timeline ─────────────────────────────────
  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !result) return
    const onTime = () => {
      const t = audio.currentTime
      setCurrentTime(t)
      if (result.spatial_timeline) {
        // Find closest timeline entry
        const entry = result.spatial_timeline.reduce((prev, curr) =>
          Math.abs(curr.time_sec - t) < Math.abs(prev.time_sec - t) ? curr : prev
        )
        setInstruments(entry.instruments || [])
      }
    }
    audio.addEventListener("timeupdate", onTime)
    return () => audio.removeEventListener("timeupdate", onTime)
  }, [result])

  // ── Download ──────────────────────────────────────────────────────────────
  const handleDownload = () => {
    window.open(API + "/download/" + jobId)
  }

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current)
    setFile(null); setJobId(null); setStatus("idle")
    setProgress(0); setMessage(""); setResult(null)
    setCurrentTime(0); setInstruments([])
    window.history.replaceState(null, "", window.location.pathname)
  }

  // ── Step labels ───────────────────────────────────────────────────────────
  const STEPS = [
    "Preprocess", "Separate", "Detect", "Motion", "Spatial", "HRTF", "Export"
  ]
  const currentStep = Math.ceil(progress / 100 * 7)

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">🎵 Raga Spatial</div>
        <div className="tagline">AI-powered spatial audio for South Indian cinematic music</div>
      </header>

      <main className="main">
        {/* Upload zone */}
        {status === "idle" && (
          <div className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}>
            <div className="drop-icon">🎶</div>
            {file ? (
              <>
                <div className="drop-filename">{file.name}</div>
                <div className="drop-size">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
                <label className="motion-toggle" style={{ display: "flex", gap: 8, alignItems: "center", margin: "8px 0", cursor: "pointer" }}>
                  <input type="checkbox" checked={motionOn} onChange={(e) => setMotionOn(e.target.checked)} />
                  Moving instruments (experimental, off by default)
                </label>
                <button className="btn-primary" onClick={handleUpload}>
                  Process Song →
                </button>
                <button className="btn-ghost" onClick={() => setFile(null)}>
                  Choose different file
                </button>
              </>
            ) : (
              <>
                <div className="drop-text">Drop your song here</div>
                <div className="drop-sub">or</div>
                <label className="btn-primary">
                  Browse files
                  <input type="file" accept=".mp3,.wav,.flac,.m4a"
                    onChange={handleFileInput} style={{ display: "none" }} />
                </label>
                <div className="drop-formats">MP3 · WAV · FLAC · M4A</div>
              </>
            )}
          </div>
        )}

        {/* Progress */}
        {(status === "uploading" || status === "running") && (
          <div className="progress-wrap">
            <div className="progress-title">Processing {file?.name}</div>
            <div className="progress-bar-outer">
              <div className="progress-bar-inner" style={{ width: progress + "%" }} />
            </div>
            <div className="progress-pct">{progress}%</div>
            <div className="progress-msg">{message}</div>
            <div className="steps-row">
              {STEPS.map((s, i) => (
                <div key={i} className={`step ${i < currentStep ? "done" : ""} ${i === currentStep - 1 ? "active" : ""}`}>
                  <div className="step-dot" />
                  <div className="step-label">{s}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="error-box">
            <div className="error-title">Pipeline error</div>
            <div className="error-msg">{message}</div>
            <button className="btn-primary" onClick={handleReset}>Try again</button>
          </div>
        )}

        {/* Results */}
        {status === "complete" && result && (
          <div className="results">
            {/* Song info strip */}
            <div className="info-strip">
              <div className="info-item">
                <span className="info-label">Duration</span>
                <span className="info-val">{result.duration_sec}s</span>
              </div>
              <div className="info-item">
                <span className="info-label">Tempo</span>
                <span className="info-val">{result.tempo_bpm} BPM</span>
              </div>
              <div className="info-item">
                <span className="info-label">Motion events</span>
                <span className="info-val">{result.motion_summary?.total_events}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Motion coverage</span>
                <span className="info-val">{result.motion_summary?.motion_coverage_pct}%</span>
              </div>
              <div className="info-item">
                <span className="info-label">Dominant instrument</span>
                <span className="info-val">{result.motion_summary?.total_events ? result.motion_summary.dominant_instrument : "none"}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Stems</span>
                <span className="info-val">
                  {result.mode === "8-stem"
                    ? result.stems8.filter((s) => s.active).length + " of " + result.stems8.length + " active"
                    : "2 (vocals + instrumental)"}
                </span>
              </div>
            </div>

            {/* 8-stem mode: zones with depth layers + live stem mixer */}
            {result.mode === "8-stem" && (
              <>
                <Stems8View api={API} jobId={jobId} result={result}
                  onTime={setCurrentTime} />
                {result.motion_events?.length > 0 && (
                  <div className="panel">
                    <div className="panel-title">Motion events</div>
                    {result.motion_events.map((ev, i) => (
                      <div key={i} className="motion-row">
                        <div className="motion-time">{ev.start_sec}s – {ev.end_sec}s</div>
                        <div className="motion-type">{ev.type}</div>
                        <div className="motion-dir">{ev.spatial_motion.replaceAll("_", " ")}</div>
                        <div className="motion-instr">{ev.dominant_instrument}</div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}

            {/* 2-stem mode: main panels */}
            {result.mode !== "8-stem" && (
            <div className="panels">
              {/* Left: spatial map */}
              <div className="panel">
                <div className="panel-title">Live spatial map</div>
                <div className="panel-sub">Instrument positions update as audio plays</div>
                <SpatialMap instruments={instruments}
                  currentTime={currentTime} />
              </div>

              {/* Right: detection */}
              <div className="panel">
                <DetectionPanel summary={result.detection_summary} />

                {/* Motion events */}
                <div className="panel-title" style={{marginTop:16}}>Motion events</div>
                {result.motion_events?.map((ev, i) => (
                  <div key={i} className="motion-row">
                    <div className="motion-time">
                      {ev.start_sec}s – {ev.end_sec}s
                    </div>
                    <div className="motion-type">{ev.type}</div>
                    <div className="motion-dir">{ev.spatial_motion.replace("_", " ")}</div>
                    <div className="motion-instr">{ev.dominant_instrument}</div>
                  </div>
                ))}
              </div>
            </div>
            )}

            {/* Timeline */}
            <TimelineBar motionEvents={result.motion_events}
              duration={result.duration_sec}
              currentTime={currentTime} />

            {/* Audio player (2-stem mode; the 8-stem view has its own transport) */}
            {result.mode !== "8-stem" && (
            <div className="audio-section">
              <div className="panel-title">Spatial audio output</div>
              <div className="audio-sub">🎧 Listen with headphones for full spatial effect</div>
              <audio ref={audioRef} controls className="audio-player"
                src={API + "/download/" + jobId} />
            </div>
            )}

            {/* Actions */}
            <div className="actions">
              <button className="btn-primary" onClick={handleDownload}>
                ⬇ Download spatial WAV
              </button>
              <button className="btn-ghost" onClick={handleReset}>
                Process another song
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

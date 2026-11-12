// Raga Spatial - 8-stem view: live spatial map with depth layers and a stem mixer.
// Every stem arrives already placed in space (binaural). They are decoded into
// memory and played together through the Web Audio API, sample-aligned, so
// volume / mute / solo act instantly without re-rendering on the server.

import { useEffect, useRef, useState } from "react"

class StemEngine {
  constructor() {
    this.ctx = null
    this.master = null
    this.buffers = {}
    this.gains = {}
    this.sources = []
    this.playing = false
    this.offset = 0
    this.startedAt = 0
    this.duration = 0
  }

  async load(api, jobId, names, onProgress) {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)()
    this.master = this.ctx.createGain()
    this.master.connect(this.ctx.destination)
    let done = 0
    await Promise.all(names.map(async (name) => {
      const res = await fetch(`${api}/stem/${jobId}/${name}`)
      if (!res.ok) throw new Error("Could not load the " + name + " stem")
      const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer())
      const gain = this.ctx.createGain()
      gain.connect(this.master)
      this.buffers[name] = buffer
      this.gains[name] = gain
      this.duration = Math.max(this.duration, buffer.duration)
      done += 1
      if (onProgress) onProgress(done, names.length)
    }))
  }

  currentTime() {
    if (!this.ctx) return 0
    if (!this.playing) return this.offset
    const t = this.offset + Math.max(0, this.ctx.currentTime - this.startedAt)
    return Math.min(t, this.duration)
  }

  play() {
    if (!this.ctx || this.playing) return
    if (this.offset >= this.duration - 0.05) this.offset = 0
    this.ctx.resume()
    const when = this.ctx.currentTime + 0.06      // one shared start time for all stems
    this.sources = Object.keys(this.buffers).map((name) => {
      const src = this.ctx.createBufferSource()
      src.buffer = this.buffers[name]
      src.connect(this.gains[name])
      src.start(when, this.offset)
      return src
    })
    this.startedAt = when
    this.playing = true
  }

  stopSources() {
    this.sources.forEach((src) => {
      try { src.stop() } catch (e) { /* already stopped */ }
      src.disconnect()
    })
    this.sources = []
  }

  pause() {
    if (!this.playing) return
    this.offset = this.currentTime()
    this.stopSources()
    this.playing = false
  }

  seek(t) {
    const wasPlaying = this.playing
    if (wasPlaying) { this.stopSources(); this.playing = false }
    this.offset = Math.max(0, Math.min(t, this.duration))
    if (wasPlaying) this.play()
  }

  setGain(name, value) {
    const gain = this.gains[name]
    if (gain) gain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.015)
  }

  setMaster(value) {
    if (this.master) this.master.gain.setTargetAtTime(value, this.ctx.currentTime, 0.015)
  }

  dispose() {
    this.stopSources()
    if (this.ctx) this.ctx.close()
    this.ctx = null
  }
}

const fmt = (t) => {
  const s = Math.max(0, Math.floor(t))
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0")
}

// ── Spatial map: 8 zones on depth rings ───────────────────────────────────────
function SpatialMap8({ stems, time, envelopeHz, audible, onSolo }) {
  const size = 400
  const c = size / 2
  const R = 160
  const radius = (distance) => R * (0.2 + 0.8 * distance)
  const toXY = (azimuth, distance) => {
    const rad = ((azimuth - 90) * Math.PI) / 180
    return { x: c + radius(distance) * Math.cos(rad), y: c + radius(distance) * Math.sin(rad) }
  }
  const idx = Math.floor(time * envelopeHz)

  return (
    <div className="spatial-map-wrap">
      <svg viewBox={`0 0 ${size} ${size}`} className="map8">
        {stems.map((s) => (
          <circle key={"ring-" + s.name} cx={c} cy={c} r={radius(s.distance)}
            fill="none" stroke={s.active ? s.color : "#334155"}
            strokeOpacity={s.active ? 0.18 : 0.12} strokeWidth={1}
            strokeDasharray={s.active ? "" : "3 4"} />
        ))}
        <text x={c} y={16} textAnchor="middle" className="map-dir">FRONT</text>
        <text x={c} y={size - 8} textAnchor="middle" className="map-dir">REAR</text>
        <text x={12} y={c + 4} textAnchor="start" className="map-dir">L</text>
        <text x={size - 12} y={c + 4} textAnchor="end" className="map-dir">R</text>
        <circle cx={c} cy={c} r={15} fill="#0f172a" stroke="#475569" strokeWidth={2} />
        <text x={c} y={c + 4} textAnchor="middle" fill="#94a3b8" fontSize={10}>you</text>

        {stems.map((s) => {
          const level = s.active ? (s.envelope[Math.min(idx, s.envelope.length - 1)] || 0) : 0
          const az = s.trajectory ? (s.trajectory[Math.min(idx, s.trajectory.length - 1)] ?? s.azimuth) : s.azimuth
          const { x, y } = toXY(az, s.distance)
          const on = s.active && audible[s.name]
          const moving = s.trajectory && Math.abs(az - s.azimuth) > 2.5
          const r = 8 + 12 * Math.min(1, level * 1.6)
          return (
            <g key={s.name} className={s.active ? "zone" : "zone off"}
              onClick={() => s.active && onSolo(s.name)}>
              {on && <circle cx={x} cy={y} r={r + 8} fill={s.color} opacity={0.12 + 0.25 * level} />}
              {moving && on && <circle cx={x} cy={y} r={r + 13} fill="none"
                stroke={s.color} strokeWidth={1.5} opacity={0.5} />}
              <circle cx={x} cy={y} r={s.active ? r : 5}
                fill={on ? s.color : "none"} stroke={s.active ? s.color : "#475569"}
                strokeWidth={1.5} opacity={on ? 0.45 + 0.55 * Math.min(1, level * 2) : 0.5} />
              <text x={x} y={y + (s.active ? r : 5) + 13} textAnchor="middle"
                fill={s.active ? s.color : "#475569"} fontSize={10}
                fontWeight={moving ? 700 : 400}>{s.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ── Main 8-stem view ──────────────────────────────────────────────────────────
export default function Stems8View({ api, jobId, result, onTime }) {
  const stems = result.stems8
  const envelopeHz = result.envelope_hz || 4
  const engineRef = useRef(null)
  const [loaded, setLoaded] = useState(0)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState("")
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [master, setMaster] = useState(1)
  const [mix, setMix] = useState(() =>
    Object.fromEntries(stems.map((s) => [s.name, { vol: 1, mute: false, solo: false }])))

  const activeNames = stems.filter((s) => s.active).map((s) => s.name)
  const anySolo = activeNames.some((n) => mix[n].solo)
  const audible = Object.fromEntries(stems.map((s) => [
    s.name, s.active && !mix[s.name].mute && (!anySolo || mix[s.name].solo)]))

  // load all active stems once
  useEffect(() => {
    const engine = new StemEngine()
    engineRef.current = engine
    let cancelled = false
    engine.load(api, jobId, activeNames, (done) => { if (!cancelled) setLoaded(done) })
      .then(() => { if (!cancelled) setReady(true) })
      .catch((e) => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true; engine.dispose() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, jobId])

  // push mixer state to the audio graph
  useEffect(() => {
    const engine = engineRef.current
    if (!engine || !ready) return
    activeNames.forEach((n) => engine.setGain(n, audible[n] ? mix[n].vol : 0))
    engine.setMaster(master)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mix, master, ready])

  // follow the playhead
  useEffect(() => {
    if (!ready) return
    let raf = 0
    let last = -1
    const tick = () => {
      const engine = engineRef.current
      if (engine) {
        const t = engine.currentTime()
        if (Math.abs(t - last) > 0.06) {
          last = t
          setTime(t)
          if (onTime) onTime(t)
        }
        if (engine.playing && t >= engine.duration - 0.02) {
          engine.pause()
          engine.offset = 0
          setPlaying(false)
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready])

  const toggle = () => {
    const engine = engineRef.current
    if (!engine || !ready) return
    if (engine.playing) { engine.pause(); setPlaying(false) } else { engine.play(); setPlaying(true) }
  }
  const seek = (t) => { const e = engineRef.current; if (e && ready) { e.seek(t); setTime(t) } }
  const update = (name, patch) => setMix((m) => ({ ...m, [name]: { ...m[name], ...patch } }))
  const resetMix = () => {
    setMix(Object.fromEntries(stems.map((s) => [s.name, { vol: 1, mute: false, solo: false }])))
    setMaster(1)
  }
  const duration = engineRef.current?.duration || result.duration_sec
  const idx = Math.floor(time * envelopeHz)

  return (
    <>
      <div className="panels panels8">
        <div className="panel">
          <div className="panel-title">Live spatial map</div>
          <div className="panel-sub">
            8 depth layers, nearest in the centre. Dots grow with each stem's level. Click a dot to solo it.
          </div>
          <SpatialMap8 stems={stems} time={time} envelopeHz={envelopeHz}
            audible={audible} onSolo={(n) => update(n, { solo: !mix[n].solo })} />
        </div>

        <div className="panel">
          <div className="panel-title">Stem mixer</div>
          <div className="panel-sub">
            {result.separation_mode === "hybrid"
              ? "Vocals from MelBandRoformer, instruments from the custom 8-stem model"
              : "All stems from the custom 8-stem model"}
            {result.model_epoch ? " (epoch " + result.model_epoch + ")" : ""}
          </div>
          {stems.map((s) => {
            const m = mix[s.name]
            const level = s.active ? (s.envelope[Math.min(idx, s.envelope.length - 1)] || 0) : 0
            return (
              <div key={s.name} className={"stem-row" + (s.active ? "" : " absent")
                + (s.active && !audible[s.name] ? " silenced" : "")}>
                <div className="stem-head">
                  <span className="stem-dot" style={{ background: s.color }} />
                  <span className="stem-name">{s.label}</span>
                  <span className="stem-layer">layer {s.layer}</span>
                  {s.active && <span className="stem-share">{s.share_pct}%</span>}
                  {s.motion_count > 0 && <span className="stem-move">moves</span>}
                </div>
                <div className="stem-detail">
                  {!s.active ? "not present in this song"
                    : s.instruments.length > 0
                      ? s.instruments.map((i) => i.label + " " + Math.round(i.avg_confidence * 100) + "%").join(" · ")
                      : s.contains}
                </div>
                {s.active && (
                  <div className="stem-controls">
                    <div className="stem-meter">
                      <div className="stem-meter-fill" style={{
                        width: Math.min(100, level * 130) + "%",
                        background: s.color, opacity: audible[s.name] ? 1 : 0.25,
                      }} />
                    </div>
                    <input type="range" min={0} max={1.5} step={0.01} value={m.vol}
                      className="stem-slider" style={{ accentColor: s.color }}
                      onChange={(e) => update(s.name, { vol: Number(e.target.value) })}
                      title={"Volume " + Math.round(m.vol * 100) + "%"} />
                    <button className={"stem-btn" + (m.mute ? " on-mute" : "")}
                      onClick={() => update(s.name, { mute: !m.mute })} title="Mute">M</button>
                    <button className={"stem-btn" + (m.solo ? " on-solo" : "")}
                      onClick={() => update(s.name, { solo: !m.solo })} title="Solo">S</button>
                    <a className="stem-btn stem-dl" title="Download this stem (dry WAV)"
                      href={`${api}/download/${jobId}/stems/${s.name}`}>⬇</a>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="audio-section">
        <div className="panel-title">Spatial audio output</div>
        <div className="audio-sub">🎧 Listen with headphones for full spatial effect</div>
        {error && <div className="error-msg">{error}</div>}
        {!ready && !error && (
          <div className="audio-sub">Loading stems {loaded}/{activeNames.length}...</div>
        )}
        <div className="transport">
          <button className="btn-primary transport-play" onClick={toggle} disabled={!ready}>
            {playing ? "❚❚ Pause" : "▶ Play"}
          </button>
          <span className="transport-time">{fmt(time)} / {fmt(duration)}</span>
          <input type="range" className="transport-seek" min={0} max={duration} step={0.05}
            value={Math.min(time, duration)} disabled={!ready}
            onChange={(e) => seek(Number(e.target.value))} />
          <span className="transport-vol-label">Master</span>
          <input type="range" className="transport-vol" min={0} max={1.5} step={0.01}
            value={master} onChange={(e) => setMaster(Number(e.target.value))} />
          <button className="btn-ghost" onClick={resetMix}>Reset mix</button>
        </div>
      </div>
    </>
  )
}

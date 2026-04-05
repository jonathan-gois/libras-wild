/**
 * skeleton.js — Renderiza esqueleto MediaPipe no canvas.
 *
 * Estrutura dos landmarks normalizados (75 pontos × 2 coords = array de 75 [x,y]):
 *   0-32:  Pose
 *   33-53: Mão esquerda
 *   54-74: Mão direita
 */

const POSE_COLOR  = "rgba(100,180,255,0.8)";
const LEFT_COLOR  = "rgba(50,220,120,0.9)";
const RIGHT_COLOR = "rgba(255,160,50,0.9)";
const JOINT_R     = 3;
const JOINT_R_HAND = 4;

// Conexões Pose (subset: torso + braços)
const POSE_CONN = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
];

// Conexões dedo (padrão: 5 dedos × 4 segmentos)
function handConns(base) {
  const fingers = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
  const conns = [];
  fingers.forEach(f => {
    for (let i = 0; i < f.length - 1; i++)
      conns.push([base + f[i], base + f[i+1]]);
  });
  // palma
  [1,5,9,13,17].forEach(i => conns.push([base+0, base+i]));
  return conns;
}

const LEFT_CONN  = handConns(33);
const RIGHT_CONN = handConns(54);

class SkeletonRenderer {
  constructor(canvas) {
    this.canvas  = canvas;
    this.ctx     = canvas.getContext("2d");
    this.frames  = [];   // array of normalized frames
    this.current = 0;
    this.animId  = null;
    this.show    = true;
    this.showHands = true;
    this._scale  = 1;
    this._ox = 0;
    this._oy = 0;
  }

  load(keyframes) {
    this.frames  = keyframes || [];
    this.current = 0;
    this.clear();
    if (this.frames.length > 0) this.drawFrame(0);
  }

  /** Calcula escala e offset para caber no canvas com padding. */
  _computeTransform(pts) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = 40;
    this._scale = Math.min((W - pad*2) / rangeX, (H - pad*2) / rangeY) * 0.65;
    this._ox = W/2 - (minX + rangeX/2) * this._scale;
    this._oy = H/2 - (minY + rangeY/2) * this._scale;
  }

  _pt(p) {
    return [p[0] * this._scale + this._ox,
            p[1] * this._scale + this._oy];
  }

  drawFrame(idx) {
    if (!this.show || !this.frames.length) { this.clear(); return; }
    const pts = this.frames[idx % this.frames.length];
    const W = this.canvas.width  = this.canvas.offsetWidth;
    const H = this.canvas.height = this.canvas.offsetHeight;
    this.ctx.clearRect(0, 0, W, H);
    this._computeTransform(pts);

    const drawConn = (conns, color) => {
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth   = 2;
      conns.forEach(([a, b]) => {
        if (!pts[a] || !pts[b]) return;
        const [ax, ay] = this._pt(pts[a]);
        const [bx, by] = this._pt(pts[b]);
        this.ctx.beginPath();
        this.ctx.moveTo(ax, ay);
        this.ctx.lineTo(bx, by);
        this.ctx.stroke();
      });
    };

    const drawJoints = (indices, color, r) => {
      this.ctx.fillStyle = color;
      indices.forEach(i => {
        if (!pts[i]) return;
        const [x, y] = this._pt(pts[i]);
        this.ctx.beginPath();
        this.ctx.arc(x, y, r, 0, Math.PI*2);
        this.ctx.fill();
      });
    };

    drawConn(POSE_CONN, POSE_COLOR);
    if (this.showHands) {
      drawConn(LEFT_CONN,  LEFT_COLOR);
      drawConn(RIGHT_CONN, RIGHT_COLOR);
    }

    // Joints
    const poseIdx = Array.from({length: 33}, (_, i) => i);
    drawJoints(poseIdx, POSE_COLOR, JOINT_R);
    if (this.showHands) {
      drawJoints(Array.from({length:21},(_,i)=>i+33), LEFT_COLOR,  JOINT_R_HAND);
      drawJoints(Array.from({length:21},(_,i)=>i+54), RIGHT_COLOR, JOINT_R_HAND);
    }

    this.current = idx;
  }

  /** Animação: percorre frames em loop. */
  animate(fps = 8) {
    this.stop();
    if (!this.frames.length) return;
    let idx = 0;
    const interval = 1000 / fps;
    let last = 0;
    const loop = (ts) => {
      if (ts - last >= interval) {
        this.drawFrame(idx);
        idx = (idx + 1) % this.frames.length;
        last = ts;
      }
      this.animId = requestAnimationFrame(loop);
    };
    this.animId = requestAnimationFrame(loop);
  }

  stop() {
    if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; }
  }

  clear() {
    const W = this.canvas.width  = this.canvas.offsetWidth || 640;
    const H = this.canvas.height = this.canvas.offsetHeight || 360;
    this.ctx.clearRect(0, 0, W, H);
  }

  setVisible(v) { this.show = v; if (!v) this.clear(); else if (this.frames.length) this.drawFrame(this.current); }
  setShowHands(v) { this.showHands = v; if (this.frames.length) this.drawFrame(this.current); }
}

// Stores exactly the server's 12 world joints. No video or model predictions are saved.
export class PoseRecorder {
  constructor(config) {
    this.config = config;
    this.records = [];
    this.active = null;
  }
  start({subject, label, split, seconds}, now) {
    if (this.active) throw new Error('이미 녹화 중입니다.');
    if (!/^[a-zA-Z0-9_-]{1,40}$/.test(subject)) throw new Error('피험자 ID는 영문/숫자/_/- 1~40자로 입력하세요.');
    if (!['idle','m01','m03','m04','m09'].includes(label) || !['train','val','test'].includes(split)) throw new Error('라벨/분할을 확인하세요.');
    if (!Number.isFinite(seconds) || seconds < 10 || seconds > 120) throw new Error('녹화 길이는 10~120초입니다.');
    this.active = {subject_id:subject, session_id:crypto.randomUUID(), label, split,
      requested_seconds:seconds, created_at:new Date().toISOString(),
      start_at:now+5000, end_at:now+5000+seconds*1000, frames:[]};
  }
  capture(now, joints) {
    const r=this.active;
    if (!r || now<r.start_at || now>=r.end_at) return;
    if (joints.length!==12 || joints.some(p=>!['x','y','z','visibility'].every(k=>Number.isFinite(p[k])))) return;
    r.frames.push({timestamp_ms:now-r.start_at,joints:joints.map(p=>({...p}))});
  }
  tick(now) {
    if (this.active && now>=this.active.end_at) return this.finish('completed');
    return null;
  }
  finish(reason='interrupted') {
    const r=this.active;
    if (!r) return null;
    this.active=null;
    const {start_at,end_at,...record}=r;
    record.completion=reason;
    record.captured_seconds=record.frames.length>1 ?
      (record.frames.at(-1).timestamp_ms-record.frames[0].timestamp_ms)/1000 : 0;
    this.records.push(record);
    return record;
  }
  bundle() {
    return {format:'pose_tcn_recordings_v1',coordinate_system:'mediapipe_world',
      joints:this.config.joints,mediapipe_indices:this.config.mediapipe_indices,
      target_fps:this.config.fps,recordings:this.records};
  }
}

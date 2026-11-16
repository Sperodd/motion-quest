import {PoseRecorder} from '/static/recorder.mjs';
import {EpisodeCounter} from '/static/counter.mjs';
import {createPoseBridge} from '/static/godot-bridge.mjs';
const godotBridge=createPoseBridge();
window.PoseTCN=Object.freeze({getSnapshotJson:()=>godotBridge.getSnapshotJson()});
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const names = {m01:'딥 스쿼트',m03:'인라인 런지',m04:'사이드 런지',m06:'서서 다리 곧게 들기',m07:'서서 팔 옆으로 들기',m09:'서서 어깨 내·외회전',idle:'대기 (idle)'};
const className = i => names[config.codes?.[i] ?? config.classes[i].split(' ')[0].toLowerCase()] ?? config.classes[i];
const edges = [[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11]];
let config, detector, socket, media, running = false, pending = false, frameId;
let lastVideo = -1, lastDetect = -Infinity, lastReply = 0, lastPrediction = 0, tickStart = 0, ticks = 0;
let rows = [];
let generation = 0;
let recorder;
let counter;
let poseMissing=false;
function renderCounts() {
  godotBridge.setCounter(counter);
  $('label').textContent=counter?.current ? (names[counter.current] ?? counter.current) : '아직 확정된 동작 없음';
  $('score').textContent=Number.isFinite(counter?.currentScore) ? `${(counter.currentScore*100).toFixed(2)}%` : '—';
  $('count-list').textContent=counter ? Object.entries(counter.counts).map(([code,n])=>`${names[code] ?? code}: ${n}회`).join(' · ') : '—';
  $('count-status').textContent=!counter ? '카메라를 시작하세요.' : !counter.enabled ? 'idle이 없는 모델: 동작 확인만 가능, 카운트 비활성화' :
    counter.phase==='moving' ? `${names[counter.active] ?? counter.active} 인정 · 현재 동작이 idle로 바뀌면 +1` :
    counter.phase==='ready' ? 'idle 인정 · 운동 대기' : '카운트를 시작하려면 idle을 먼저 확정하세요.';
  $('count-detail').textContent=counter ? `${counter.reason} | 후보: ${names[counter.candidate] ?? '—'} | 연속 확인: ${(counter.progressMs/1000).toFixed(2)} / 1.00초` : '';
}
function interruptCount(reason='카메라/연결 종료 · idle부터 시작') {counter?.interrupt(reason);renderCounts();}
function breakConfirmation(reason) {counter?.breakCandidate(reason);renderCounts();}
$('count-reset').onclick=()=>{counter?.reset();renderCounts();};
function recordingControls() {
  const active=Boolean(recorder?.active);
  $('record-start').disabled=!running || active;
  $('record-cancel').disabled=!active;
  $('record-download').disabled=active || !recorder?.records.length;
  ['record-subject','record-label','record-split','record-seconds'].forEach(id=>$(id).disabled=active);
}
function renderRecordings() {
  $('record-list').replaceChildren();
  recorder?.records.forEach(record=>{
    const li=document.createElement('li');
    li.append(document.createTextNode(`${record.subject_id} / ${record.label} / ${record.split} · ${record.captured_seconds.toFixed(1)}초 · ${record.frames.length}프레임 · ${record.completion==='completed'?'완료':'중단됨 (삭제 후 재녹화)'} `));
    const remove=document.createElement('button'); remove.textContent='목록에서 삭제'; remove.className='secondary';
    remove.disabled=Boolean(recorder.active);
    remove.onclick=()=>{recorder.records=recorder.records.filter(r=>r!==record);renderRecordings();};
    li.append(remove); $('record-list').append(li);
  });
  recordingControls();
}
$('record-start').onclick=()=>{
  try {
    recorder.start({subject:$('record-subject').value.trim(),label:$('record-label').value,
      split:$('record-split').value,seconds:Number($('record-seconds').value)},performance.now());
    renderRecordings();
  } catch(error) {$('record-status').textContent=error.message;}
};
$('record-cancel').onclick=()=>{
  recorder.finish();renderRecordings();$('record-status').textContent='중단한 녹화는 목록에서 삭제한 뒤 다시 녹화하세요.';
};
$('record-download').onclick=()=>{
  const blob=new Blob([JSON.stringify(recorder.bundle())],{type:'application/json'});
  const url=URL.createObjectURL(blob), a=document.createElement('a');
  a.href=url;a.download='webcam_recordings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  $('record-status').textContent='다운로드 파일을 보관하세요. 같은 녹화를 포함한 파일을 중복으로 학습에 넣지 마세요.';
};
function clearPrediction() {
  godotBridge.clearPose();
  $('raw-prediction').textContent='실시간 모델 예측: —'; $('latency').textContent = '—';
  rows.forEach(row => { row.bar.style.width = '0%'; row.score.textContent = '—'; });
  lastPrediction = 0;
}
function stop(reason = '중지되었습니다. 다시 시작할 수 있습니다.') {
  interruptCount();
  if (recorder?.active) {recorder.finish();$('record-status').textContent='카메라 종료로 녹화가 중단되었습니다. 해당 녹화를 삭제하고 재시도하세요.';}
  generation++;
  running = false; cancelAnimationFrame(frameId);
  socket?.close(); socket = null;
  media?.getTracks().forEach(track => track.stop()); media = null; video.srcObject = null;
  pending = false; clearPrediction(); ctx.clearRect(0,0,canvas.width,canvas.height);
  godotBridge.setConnection(false);
  $('progress').value = 0; $('status').textContent = reason; $('connection').textContent = '연결 종료';
  $('placeholder').hidden = false; $('start').disabled = false; $('stop').disabled = true; $('fps').textContent = '— FPS';
  renderRecordings();
}
function handleReply(event) {
  pending = false;;
  lastReply = performance.now();
  const data = JSON.parse(event.data);
  $('progress').value = data.progress ?? 0;
  if (data.status === 'prediction') {
    if (poseMissing) return;
    godotBridge.setPrediction(data);
    // The same received predictions shown below drive the one-second confirmation.
    counter?.push(data.class_id,data.score,lastReply);
    renderCounts();
    lastPrediction = lastReply;
    $('raw-prediction').textContent=`실시간 모델 예측: ${names[data.class_id] ?? data.class_id} ${(data.score*100).toFixed(2)}% (확정 전)`;
    $('latency').textContent = `${data.inference_ms.toFixed(0)} ms`;
    $('status').textContent = '현재 동작은 70% 이상 · 1초 연속 확인한 결과입니다.';
    rows.forEach((row,i) => { const value = (data.probabilities[i]*100).toFixed(1); row.bar.style.width = `${value}%`; row.score.textContent = `${value}%`; });
  } else if (data.status !== 'collecting') {
    breakConfirmation(`서버 상태: ${data.status} · 후보 확인 초기화`);
    clearPrediction();
    godotBridge.clearPose(data.status);
    $('status').textContent = data.status === 'warming_up' ? '동작을 수집 중입니다. 같은 동작을 계속해 주세요.' :
      data.status === 'invalid_input' ? `입력 확인: ${data.message}` : '전신이 잘 보이도록 서 주세요. 다시 수집합니다.';
  }
}
function send(message) {
  if (!pending && socket?.readyState === WebSocket.OPEN && socket.bufferedAmount < 16000) {
    socket.send(JSON.stringify({version:1,...message})); pending = true; lastReply = performance.now();
    return true;
  }
  return false;
}
function loop(now) {
  if (!running) return;
  if(recorder?.active) {
    const completed=recorder.tick(now);
    if(completed) {renderRecordings();$('record-status').textContent=`녹화 완료: ${completed.frames.length}프레임. 다음 녹화 전에 위치나 팔 모양을 조금 바꾸세요.`;}
    else {
      const r=recorder.active;
      $('record-status').textContent=now<r.start_at ? `준비 ${Math.ceil((r.start_at-now)/1000)}초 · 전신이 보이게 서세요.` :
        `${r.label} / ${r.split} 녹화 중 · ${Math.ceil((r.end_at-now)/1000)}초 남음 · ${r.frames.length}프레임`;
    }
  }
  if (pending && now-lastReply > 5000) { stop('서버 응답이 지연되었습니다. 다시 시작해 주세요.'); return; }
  if (lastPrediction && now-lastPrediction > 1500) { breakConfirmation('새 예측 없이 1.5초 경과 · 마지막 확정 동작 유지');clearPrediction(); $('status').textContent = '새 동작 결과를 기다립니다.'; }
  try {
    if (video.readyState >= 2 && video.currentTime !== lastVideo && now-lastDetect >= 1000/config.fps-2) {
      lastVideo = video.currentTime; lastDetect = now;
      const result = detector.detectForVideo(video, now);
      const world = result.worldLandmarks[0], image = result.landmarks[0];
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      if (world && config.mediapipe_indices.every(i => world[i])) {
        poseMissing=false;
        if (image && config.mediapipe_indices.every(i => image[i])) {
        const points = config.mediapipe_indices.map(i => image[i]);
        ctx.strokeStyle = '#71e5bf'; ctx.lineWidth = 3;
        edges.forEach(([a,b]) => { ctx.beginPath(); ctx.moveTo(points[a].x*canvas.width,points[a].y*canvas.height); ctx.lineTo(points[b].x*canvas.width,points[b].y*canvas.height); ctx.stroke(); });
        }
        const joints=config.mediapipe_indices.map(i => ({x:world[i].x,y:world[i].y,z:world[i].z,visibility:image?.[i]?.visibility ?? world[i].visibility ?? 0}));
        recorder?.capture(now,joints);
        send({type:'frame',coordinate_system:'mediapipe_world',timestamp_ms:now,joints});
      } else {
        clearPrediction();
        godotBridge.clearPose('tracking_lost');
        poseMissing=true;
        breakConfirmation('관절 좌표 없음 · 후보 확인 초기화, 마지막 확정 동작 유지');
        $('status').textContent = '관절 좌표가 없습니다. 기존 버퍼를 유지하며 기다립니다.';
        renderCounts();
      }
      ticks++;
      if (now-tickStart > 1000) { $('fps').textContent = `${(ticks*1000/(now-tickStart)).toFixed(0)} FPS`; tickStart = now; ticks = 0; }
    }
  } catch (error) { stop(`포즈 추정 오류: ${error.message}`); return; }
  frameId = requestAnimationFrame(loop);
}
$('start').onclick = async () => {
  const attempt = ++generation;
  $('start').disabled = true; $('status').textContent = 'MediaPipe Lite와 서버를 준비하고 있습니다…';
  $('stop').disabled = false;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('localhost 또는 HTTPS에서 실행해 주세요.');
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error('서버 설정을 읽지 못했습니다.');
    config = await response.json();
    if (attempt !== generation) return;
    const codes=config.codes ?? config.classes.map(label=>label.split(' ')[0].toLowerCase());
    if (!counter || JSON.stringify(counter.codes)!==JSON.stringify(codes.filter(c=>c!=='idle')) || counter.enabled!==codes.includes('idle')) counter=new EpisodeCounter(codes);
    interruptCount();poseMissing=false;
    if (!recorder) recorder=new PoseRecorder(config);
    $('class-count').textContent=config.classes.length;
    $('model-info').textContent = config.feature_schema === 'positions_angles_v1'
      ? '위치 + 각도 모델 · 54개 특징'
      : config.feature_schema === 'joint_angles_v1'
        ? '이전 각도 모델 · 18개 특징 (통합 모델 재학습 전)'
        : '이전 위치 모델 · 36개 특징 (통합 모델 재학습 전)';
    {
      $('classes').replaceChildren();
      rows = config.classes.map((label,i) => {
        const element = document.createElement('div'); element.className = 'class-row';
        element.innerHTML = `<div class="class-title"><span></span><span class="value">—</span></div><div class="track"><div class="fill"></div></div>`;
        element.querySelector('span').textContent = `${label.split(' ')[0]} · ${className(i)}`;
        $('classes').append(element); return {bar:element.querySelector('.fill'),score:element.querySelector('.value')};
      });
    }
    if (!detector) {
      const {FilesetResolver, PoseLandmarker} = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs');
      if (attempt !== generation) return;
      const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
      if (attempt !== generation) return;
      const options = {baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',delegate:'GPU'},runningMode:'VIDEO',numPoses:1};
      let created;
      try { created = await PoseLandmarker.createFromOptions(vision,options); }
      catch { options.baseOptions.delegate = 'CPU'; created = await PoseLandmarker.createFromOptions(vision,options); }
      if (attempt !== generation) { created.close(); return; }
      detector = created;
    }
    $('status').textContent = '카메라 권한을 허용해 주세요. 응답이 없으면 중지 후 Chrome 또는 Edge에서 다시 시도하세요.';
    const acquired = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}},audio:false});
    if (attempt !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
    media = acquired;
    video.srcObject = media; await video.play();
    if (attempt !== generation) return;
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/pose`);
    await new Promise((resolve,reject) => {
      const timeout = setTimeout(() => reject(new Error('서버 연결 시간 초과')),5000);
      socket.onopen = () => { clearTimeout(timeout); resolve(); };
      socket.onerror = () => { clearTimeout(timeout); reject(new Error('WebSocket 연결 실패')); };
    });
    if (attempt !== generation) return;
    socket.onmessage = handleReply;
    socket.onclose = () => { if (running) stop('서버 연결이 끊겼습니다. 다시 시작해 주세요.'); };
    media.getVideoTracks()[0].onended = () => stop('카메라가 종료되었습니다.');
    running = true; pending = false; lastVideo = -1; lastDetect = -Infinity;
    godotBridge.setConnection(true);
    recordingControls();$('record-status').textContent='준비되면 녹화 시작을 누르세요. 5초 뒤 기록을 시작합니다.';
    tickStart = performance.now(); ticks = 0;
    $('connection').textContent = '실시간 연결됨'; $('stop').disabled = false; $('placeholder').hidden = true;
    $('status').textContent = '전신이 보이도록 서서 동작을 시작해 주세요.';
    frameId = requestAnimationFrame(loop);
  } catch (error) { if (attempt === generation) stop(`시작하지 못했습니다: ${error.message}`); }
};
$('stop').onclick = () => stop();
$('godot-load').onclick=()=>{
  $('godot-frame').src='/static/godot/Motion%20Quest.html';
  $('godot-frame').hidden=false;
};
document.addEventListener('visibilitychange', () => { if (document.hidden && $('start').disabled) stop('탭이 숨겨져 카메라를 중지했습니다.'); });
window.addEventListener('pagehide', () => stop());

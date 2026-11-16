// Godot and the UI share the same confirmed movement and counter snapshot.
export function createPoseBridge(now=()=>performance.now()) {
  let connected=false, raw=null, receivedAt=null, status='stopped';
  let counter={enabled:false,phase:'waiting',active:null,current:null,counts:{}};
  return {
    setConnection(value) {
      connected=value;raw=null;receivedAt=null;status=value?'warming_up':'stopped';
    },
    setPrediction(data) {
      raw={class_id:data.class_id,label:data.label,score:data.score,
        probabilities:[...data.probabilities],timestamp_ms:data.timestamp_ms};
      receivedAt=now();status='prediction';
    },
    clearPose(reason='collecting') {raw=null;receivedAt=null;status=reason;},
    setCounter(value) {
      counter=value ? {enabled:value.enabled,phase:value.phase,active:value.active,
        current:value.current ?? null,current_score:value.currentScore ?? null,
        confirmed_at_ms:value.confirmedAt ?? null,candidate:value.candidate ?? null,
        confirmation_ms:value.progressMs ?? 0,counts:{...value.counts}} :
        {enabled:false,phase:'waiting',active:null,current:null,counts:{}};
    },
    getSnapshotJson() {
      const age=receivedAt===null?null:Math.max(0,now()-receivedAt);
      const validRaw=connected && raw!==null && age<=1500;
      const pose=connected && counter.current ? {class_id:counter.current,label:counter.current,
        score:counter.current_score,timestamp_ms:counter.confirmed_at_ms} : null;
      return JSON.stringify({version:1,connected,status:raw && !validRaw && connected?'stale':status,
        pose,raw_prediction:validRaw?raw:null,pose_age_ms:pose?Math.max(0,now()-pose.timestamp_ms):null,
        raw_age_ms:age,counter});
    },
  };
}

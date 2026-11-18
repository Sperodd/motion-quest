"""Shared test fixtures; no test cases are imported by other tests."""

def frame(t):
    # Synthetic standing skeleton in MediaPipe world coordinates (not an accuracy test).
    points = [[-.2,-.6,0],[.2,-.6,0],[-.3,-.3,0],[.3,-.3,0],
              [-.3,0,0],[.3,0,0],[-.15,0,0],[.15,0,0],
              [-.15,.5,0],[.15,.5,0],[-.15,1,0],[.15,1,0]]
    return {'version':1,'type':'frame','coordinate_system':'mediapipe_world','timestamp_ms':t,
            'joints':[dict(zip(['x','y','z'],p),visibility=1.) for p in points]}



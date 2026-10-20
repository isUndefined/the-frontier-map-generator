"""Verify native pixel placement using landmarks; never transform an image."""
import cv2
import numpy as np


def verify_identity(generated,reference,fixed):
    limits={'minimum_inliers':60,'minimum_inlier_fraction':.85,
            'maximum_displacement_p90_px':2.,'maximum_fitted_drift_px':2.,
            'minimum_context_span_fraction':.60}
    report={'method':'native_landmark_identity','status':'rejected','registration_applied':False,'limits':limits}
    detector=cv2.SIFT_create(nfeatures=5000)
    mask=np.where(fixed,255,0).astype(np.uint8)
    ka,da=detector.detectAndCompute(cv2.cvtColor(generated,cv2.COLOR_RGB2GRAY),mask)
    kb,db=detector.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),mask)
    if da is None or db is None or min(len(ka),len(kb))<60:
        return {**report,'reason':'Insufficient native landmarks','failed_checks':['minimum_landmarks']}
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
    if len(good)<60:return {**report,'reason':'Insufficient unambiguous matches','failed_checks':['minimum_matches']}
    src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if matrix is None or inliers is None:return {**report,'reason':'Inconsistent landmarks','failed_checks':['consistent_transform']}
    inside=inliers.ravel()!=0
    ys,xs=np.nonzero(fixed)
    corners=np.float32([[xs.min(),ys.min()],[xs.max(),ys.min()],
                        [xs.max(),ys.max()],[xs.min(),ys.max()]])
    drift=np.linalg.norm(corners@matrix[:,:2].T+matrix[:,2]-corners,axis=1)
    distances=np.linalg.norm(dst-src,axis=1)
    span=np.ptp(src[inside],axis=0)
    area=np.array([max(1,np.ptp(xs)),max(1,np.ptp(ys))])
    coverage=span/area
    # Only an identity-sized result is accepted. Fitting estimates drift for
    # rejection; it is NEVER used to warp, resize, rotate or shift any pixels.
    tests = [
        ('minimum_inliers',int(inside.sum())>=60),
        ('minimum_inlier_fraction',float(inside.mean())>=.85),
        ('maximum_displacement_p90_px',float(np.quantile(distances,.9))<=2.),
        ('maximum_fitted_drift_px',float(drift.max())<=2.),
        ('minimum_context_span_fraction',bool(np.all(coverage>=.60))),
    ]
    failed=[name for name,passed in tests if not passed]
    passed=not failed
    return {**report,'status':'accepted' if passed else 'rejected',
            'matches':len(good),'inliers':int(inside.sum()),
            'inlier_fraction':float(inside.mean()),
            'max_fitted_drift_px':float(drift.max()),
            'landmark_displacement_p90_px':float(np.quantile(distances,.9)),
            'context_span_fraction':coverage.tolist(),'failed_checks':failed,
            'reason':'Native landmarks remain within 2px; exact locks restored from sources' if passed else 'Context geometry moved or landmarks are inconsistent: '+', '.join(failed)}

'''
OpenCV色块立方体上表面角点检测与位姿估计
--------------------------------------------------
- 作者: 阿凯
- Email: kyle.xing@fashionstar.com.hk
- 更新时间: 2020-03-11
--------------------------------------------------
'''

import cv2
import logging
import numpy as np
from cv_util import find_contours,find_max_contour
from cv_camera import Camera
from cubic_status import CubicStatus
from geometry import *
from config import *

def find_cubic_contour(img:np.array, bgr_lowerb_list:list, bgr_upperb_list:list,\
            min_width:int=COLOR_CUBIC_CONTOUR_MIN_WIDTH,\
            min_height:int=COLOR_CUBIC_CONTOUR_MIN_HEIGHT,\
            max_width:int=COLOR_CUBIC_CONTOUR_MAX_WIDTH,\
            max_height:int=COLOR_CUBIC_CONTOUR_MAX_HEIGHT,\
            canvas:np.array=None,roi:tuple=None):
    '''找到物块上表面的连通域(默认一次只找一个)'''

    # bin_canny = cv2.Canny(img,40,100) # 使用Canny算子获取图像边缘
    # bin_canny = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #1.图像二值化
    # 注: bgr_lowerb与bgr_upperb是一组二值化图像
    img_bin = None
    use_roi = roi is not None
    roi_x = None 
    roi_y = None
    roi_w = None
    roi_h = None
    # print('ROI')
    # print(roi)
    if use_roi:
        # print('USE ROI')
        roi_x, roi_y, roi_w, roi_h = roi
        img_bin =  np.zeros((roi_h, roi_w), dtype=np.uint8)
    else:
        img_bin = np.zeros(img.shape[:2], dtype=np.uint8)

    for i in range(len(bgr_lowerb_list)):
        bgr_lowerb = bgr_lowerb_list[i]
        bgr_upperb = bgr_upperb_list[i]
        if use_roi:
            img_bin = cv2.bitwise_or(img_bin, cv2.inRange(img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], lowerb=bgr_lowerb, upperb=bgr_upperb))
        else:
            img_bin = cv2.bitwise_or(img_bin, cv2.inRange(img, lowerb=bgr_lowerb, upperb=bgr_upperb))
    
    # back_bin = cv2.inRange(img, lowerb=BLACK_BACKGROUND_LOWERB, upperb=BLACK_BACKGROUND_UPPERB)
    # back_bin = cv2.bitwise_not(back_bin)
    # img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((7, 7)))

    #2.寻找满足尺寸要求的大的连通域
    ret, result = find_max_contour(img_bin, min_width, min_height, max_width, max_height)
    if not ret:
        return False, None
    (cubic_rect, contour) = result
    x0, y0, w, h = cubic_rect # 解包矩形
    
    roi_img=None

    if use_roi:
        roi_img = img[roi_y+y0:roi_y+y0+h, roi_x+x0:roi_x+x0+w] # 获取ROI图像 
    else:
        roi_img = img[y0:y0+h, x0:x0+w] # 获取ROI图像

    roi_bin = img_bin[y0:y0+h, x0:x0+w] # 获取二值化图像
    
    #3.连通域分割
    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    ret,roi_bin = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    roi_img_blur = cv2.medianBlur(roi_img, 15) # 对图像先做一下中值滤波
    bin_canny = cv2.Canny(roi_img_blur, 40,100) # 使用Canny算子获取图像边缘
    bin_canny = cv2.morphologyEx(bin_canny, cv2.MORPH_DILATE, np.ones((7, 7)), iterations=1)
    # 进行开运算,去掉内部纹理的影响
    # 将二值化图像扣去边缘部分
    roi_bin = cv2.bitwise_and(cv2.bitwise_not(bin_canny), roi_bin)
    
    if canvas is not None:
        # 在画布中绘制二值化区域
        mask = np.zeros_like(roi_img)
        mask[:,:] = [0, 200, 0]
        mask = cv2.bitwise_and(mask, mask, mask=roi_bin)
        if use_roi:
            canvas[roi_y+y0:roi_y+y0+h, roi_x+x0:roi_x+x0+w] = cv2.addWeighted(roi_img, 0.6, mask, 0.4, 0)
        else:
            canvas[y0:y0+h, x0:x0+w] = cv2.addWeighted(roi_img, 0.6, mask, 0.4, 0)
    #4.获取最大的连通域
    offset = (x0, y0) if not use_roi else (x0+roi_x, y0+roi_y)
    ret, result = find_max_contour(roi_bin, min_width, min_height, max_width, max_height,offset=offset)
    
    # 添加上ROI偏移量的信息
    # if ret and use_roi:
    #     (global_rect, contour) = result
    #     gx, gy, gw, gh = global_rect
    #     contour[:, 0, 0] += roi_x
    #     contour[:, 0, 0] += roi_y
    #     return True, ((gx+roi_x, gx+roi_y, gw, gh), contour)
    return ret, result

def points_near_segment_index(A, B, points, R):
    '''筛选距离线段距离小于50的点(只是返回索引)'''
    index_list = []
    for idx, P in enumerate(points):
        # print("idx: {} P: {}".format(idx, P))
        d = distance_point2segment(A, B, P)
        # print(d)
        if d <= R:
            index_list.append(idx)
    return np.int64(index_list)

def line_refit(A, B, points, R=15, canvas=None):
    '''直线重新拟合'''
    pts_ab_idx = points_near_segment_index(A, B, points, R=R)
    pts_ab_x = points[pts_ab_idx][:, 0]
    pts_ab_y = points[pts_ab_idx][:, 1]
    # 拟合直线
    k_AB, b_AB = np.polyfit(pts_ab_x, pts_ab_y, deg=1)
    
    x_left = int(min(A[0], B[0]) - 200)
    x_right = int(max(A[0], B[0]) + 200)
    y_left = int(k_AB * x_left + b_AB)
    y_right = int(k_AB * x_right + b_AB)
    
    if canvas is not None:
        canvas = cv2.line(canvas, (x_left, y_left), (x_right, y_right), color=(255, 255, 0), thickness=1)
    
    return (x_left, y_left),  (x_right, y_right), (k_AB, b_AB)

def get_cubic_corner(contour, cubic_rect=None, canvas=None, blob_color=(0, 0, 255)):
    '''重新计算物块上表面的交点'''
    # 获取contour的凸包(防止边缘/内部孔洞的影响)
    # 但是这样也会因为边缘出的一些未处理干净的小突出带来非常大的干扰
    contour_hull = cv2.convexHull(contour)
    # 多边形近似
    # epsilon = 0.12*cv2.arcLength(contour,True)
    # 返回点集，格式与contours相同
    # approx_cnt = cv2.approxPolyDP(contour,epsilon,True)
    # 使用外接矩形作为筛选依据 (假设现在不考虑空间)
    min_area_rect = cv2.minAreaRect(contour_hull)
    approx_cnt =  cv2.boxPoints(min_area_rect)
    n_point = len(approx_cnt)
    approx_cnt = approx_cnt.reshape((n_point, 1, 2))
    if len(approx_cnt) != 4:
        # logging.info("Error,　多变形拟合的角点个数为: {}".format(approx_cnt))
        # logging.info(approx_cnt)
        return False, None
    # 赋值近似四边形的四个定点
    D1, C1, B1, A1 =  [pt[0] for pt in approx_cnt]
    # 获取样本数据
    points = contour.reshape([-1, 2]).astype('float32')

    # 线段(直线)重新拟合
    try:
        AB_p1, AB_p2, (AB_k, AB_b) = line_refit(A1, B1, points)
        BC_p1, BC_p2, (BC_k, BC_b) = line_refit(B1, C1, points)
        CD_p1, CD_p2, (CD_k, CD_b) = line_refit(C1, D1, points)
        DA_p1, DA_p2, (DA_k, DA_b) = line_refit(D1, A1, points)
        # 计算直线与直线之间的交点作为新的角点
    
        ret1, A2 = line_cross_pt(DA_k, DA_b, AB_k, AB_b)
        ret2, B2 = line_cross_pt(AB_k, AB_b, BC_k, BC_b)
        ret3, C2 = line_cross_pt(BC_k, BC_b, CD_k, CD_b)
        ret4, D2 = line_cross_pt(CD_k, CD_b, DA_k, DA_b)
    except ValueError as ve:
        # 数据样本点太少的时候就会出现PolyFit错误
        return False, None

    if not (ret1 and ret2 and ret3 and ret4):
        # 没有交点, 数据异常
        return False, None
    # 判断A2, B2, C2, D2是否在原来的最小面积外接矩形的范围内
    # 如果超出范围就赋值为原来的值
    convex_pts = np.float32([list(A1), list(B1), list(C1), list(D1)])
    if not is_point_in_convex(convex_pts, A2):
        A2 = tuple(A1)
    if not is_point_in_convex(convex_pts, B2):
        B2 = tuple(B1)
    if not is_point_in_convex(convex_pts, C2):
        C2 = tuple(C1)
    if not is_point_in_convex(convex_pts, D2):
        D2 = tuple(D1)
    
    # 拟合四边形质量检查
    if cubic_rect is not None:
        x0, y0, w, h = cubic_rect 
        contour_hull[:, 0, 0] -= x0
        contour_hull[:, 0, 1] -= y0
        # print("Contour Hull")
        # print(contour_hull)
        # 绘制原来的图像的连通域的二值化图像
        bin_cnt_raw = np.uint8(np.zeros((h, w)))
        bin_cnt_raw = cv2.drawContours(bin_cnt_raw, contours=[contour_hull], contourIdx=0, color=255, thickness=-1)
        # 原始contour的像素个数
        pnum_cnt_raw = len(np.nonzero(bin_cnt_raw)[0])
        # 拟合得到的四边形的四个角点
        contour_fit = np.int32([A2, B2, C2, D2]).reshape((4, 1, 2))
        contour_fit[:, 0, 0] -= x0
        contour_fit[:, 0, 1] -= y0
        # print("Contour Fit")
        # print(contour_fit)
        bin_cnt_fit = np.uint8(np.zeros((h, w)))
        bin_cnt_fit = cv2.drawContours(bin_cnt_fit, contours=[contour_fit], contourIdx=0, color=255, thickness=-1)
        # bin_cnt_fit = cv2.drawContours(bin_cnt_fit, contours=[contour_fit], contourIdx=0, color=255, thickness=-1)
        # bin_cnt_fit的像素个数
        pnum_cnt_fit = len(np.nonzero(bin_cnt_fit)[0])

        # 计算相似度
        ratio = abs(pnum_cnt_fit-pnum_cnt_raw) / pnum_cnt_raw
        if  ratio > 0.06:
            # 误差大于10% 有10%的区域没有被覆盖到, 说明拟合点过于靠里
            # logging.info
            # print('相似度＝{:.2f}% 差异大于 {:.2f}%, 选用外接矩形'.format(ratio*100, RECT_FIT_QUALITY_RATIO*100))
            A2 = tuple(A1)
            B2 = tuple(B1)
            C2 = tuple(C1)
            D2 = tuple(D1)
        else:
            # print('相似度＝{:.2f}% '.format(ratio*100))
            pass
    # 计算线段交点
    ret5, O = line_cross_pt2(A2, C2, B2, D2)
    if not ret5:
        return False, None

    # 在画布上绘制四个新角点
    if canvas is not None:
        radius = 10
        white = (255, 255, 255)
        line_thickness = 3
        # 绘制边界线
        canvas = cv2.line(canvas, A2, B2, color=white, thickness=line_thickness)
        canvas = cv2.line(canvas, B2, C2, color=white, thickness=line_thickness)
        canvas = cv2.line(canvas, C2, D2, color=white, thickness=line_thickness)
        canvas = cv2.line(canvas, D2, A2, color=white, thickness=line_thickness)
        # 绘制交叉线
        canvas = cv2.line(canvas, A2, C2, color=white, thickness=line_thickness)
        canvas = cv2.line(canvas, B2, D2, color=white, thickness=line_thickness)
        # 绘制四个角点
        cv2.circle(canvas, A2, radius, color=white, thickness=-1)
        cv2.circle(canvas, B2, radius, color=white, thickness=-1)
        cv2.circle(canvas, C2, radius, color=white, thickness=-1)
        cv2.circle(canvas, D2, radius, color=white, thickness=-1)
        # 绘制中心
        cv2.circle(canvas, O, radius*5, color=white, thickness=-1)
        cv2.circle(canvas, O, radius*4, color=blob_color, thickness=-1)
    return True, (A2, B2, C2, D2, O)

def cubic_pose_estimate(camera, cubic_img_pts, canvas=None):
    '''立方体位姿估计'''
    
    r = CUBIC_SIZE/2.0 
    # 立方体坐标系下, 上表面角点与中心点的坐标
    # 立方体的坐标系原点定义在上表面上.
    cubic_obj_pts = np.float32([
        [r, r, 0], # A点
        [r, -r, 0], # B点 
        [-r, -r, 0], # C点
        [-r, r, 0], # D点
    ])

    pa, pb, pc, pd = cubic_img_pts # 立方体上表面角点在图像中的坐标
    # 立方体上表面在图像坐标系下的坐标
    
    cubic_img_pts = np.float32([list(pa), list(pb), list(pc), list(pd)]).reshape((4, 1, 2))

    # 求解旋转向量与平移向量
    retval, rotation_vec, translation_vec = cv2.solvePnP(cubic_obj_pts, cubic_img_pts, camera.intrinsic, camera.distortion, flags=cv2.SOLVEPNP_AP3P)
    # retval, rotation_vec, translation_vec = cv2.solvePnP(cubic_obj_pts, cubic_img_pts, camera.intrinsic, camera.distortion, flags=cv2.SOLVEPNP_EP3P)

    if canvas is not None:
        # 在图像中绘制坐标轴 
        axis_len = CUBIC_SIZE * 1.5 # 坐标轴的长度 单位cm
        # 立方体坐标系坐标轴的向量定义
        axis_obj_pts = np.float32([
            [0, 0, 0], # 物块坐标系原点
            [axis_len, 0, 0], # X轴
            [0, axis_len, 0], # Y轴
            [0, 0, axis_len]]) # Z轴
        # 3D点重投影到图像平面上
        axis_img_pts, _ = cv2.projectPoints(axis_obj_pts, rotation_vec, translation_vec, camera.intrinsic, camera.distortion)
        axis_img_pts = tuple(map(tuple, axis_img_pts.reshape(4, 2)))
        po, px, py, pz = axis_img_pts
        axis_thickness = 10 # 坐标轴的像素宽度
        canvas = cv2.line(canvas, po, px, (0, 0, 180), thickness=axis_thickness) # 绘制X轴
        canvas = cv2.line(canvas, po, py, (0, 180, 0), thickness=axis_thickness) # 绘制Y轴
        canvas = cv2.line(canvas, po, pz, (180, 0, 0), thickness=axis_thickness) # 绘制Z轴    

    return rotation_vec, translation_vec

def update_cubic_stats(camera:Camera, cubic_stats:dict, frame:np.array, canvas:np.array):
    '''更新立方体的状态'''
    # 遍历所有颜色
    for cname in COLOR_NAMES:
        has_cnt = False
        result = None

        lowerb = BGR_THRESHOLDS[cname][0] # 阈值下界
        upperb = BGR_THRESHOLDS[cname][1] # 阈值上界
        if cubic_stats[cname].cnt > 0:
            # 在上次的Cubic附近检索
            R = 300
            # 合法化
            roi_x, roi_y, roi_w, roi_h = (cubic_stats[cname].O[0]-R, cubic_stats[cname].O[1]-R, 2*R, 2*R)
            if roi_x < 0: roi_x = 0
            if (roi_x+roi_w) >= CAM_IMG_WIDTH: roi_w = CAM_IMG_WIDTH-roi_x-1
            if roi_y < 0: roi_y = 0
            if (roi_y+roi_h) >= CAM_IMG_HEIGHT: roi_h = CAM_IMG_HEIGHT-roi_y-1
            roi = (roi_x, roi_y, roi_w, roi_h)
            # print('{} center ({}, {})'.format(cname, cubic_stats[cname].x, cubic_stats[cname].y))
            # print('ROI = {}'.format(roi))
            has_cnt,result = find_cubic_contour(frame, lowerb, upperb, canvas=canvas, roi=roi)
            # print('ROI Search : {} cname ={}'.format(has_cnt, cname))
        # 在ROI区域内没有检索到,重新检索
        if not has_cnt:
            has_cnt,result = find_cubic_contour(frame, lowerb, upperb, canvas=canvas)
        
        # 没有合法的连通域
        if not has_cnt:
            cubic_stats[cname].update(False, None)
            continue
        (cubic_rect, contour) = result
        has_corner, result2 = get_cubic_corner(contour,cubic_rect=cubic_rect, canvas=canvas, blob_color=BGR_CANVAS_COLOR[cname])
        # 没有拟合到四边形的四个角点
        if not has_corner:
            cubic_stats[cname].update(False, None)
            continue
        # 提取角点
        A, B, C, D, O = result2
        # 位姿估计
        cubic_img_pts = [A, B, C, D]
        rvec, tvec = cubic_pose_estimate(camera, cubic_img_pts, canvas=canvas)
        # 提取为工作台的坐标
        # 将平移向量tvec,　变换为工作台的xy坐标
        wp_y, wp_x = tvec.reshape(-1)[:2]
        # 更新立方体的状态
        cubic_stats[cname].update(True, (wp_x, wp_y), O)

def display_cubic_stats(cubic_stats, canvas):
    '''绘制色块的的状态信息'''
    for cname in COLOR_NAMES:
        cubic_stat = cubic_stats[cname]
        if cubic_stat.cnt == 0: continue
        # 绘制红色色块在工作台上的坐标
        tag = "{} ({:4.1f}, {:4.1f})".format(cname, cubic_stat.x, cubic_stat.y)
        font = cv2.FONT_HERSHEY_SIMPLEX # 选择字体
        cv2.putText(canvas, text=tag, org=BGR_CANVAS_TAG_POSI[cname],\
            fontFace=font, fontScale=2, thickness=4, \
            lineType=cv2.LINE_AA, color=BGR_CANVAS_COLOR[cname])


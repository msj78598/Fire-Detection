import streamlit as st
import torch
import cv2
from PIL import Image
from datetime import datetime
import pandas as pd
import time  # لاستيراد time لتأخير الوميض

# شريط جانبي للإدارة
st.sidebar.title("الإدارة")

# إضافة خيار لإصدار تقرير Excel
st.sidebar.subheader("إصدار تقرير")

# تحديد الفترة الزمنية لاستخراج التقرير
start_date = st.sidebar.date_input("تاريخ البداية")
end_date = st.sidebar.date_input("تاريخ النهاية")

# زر لاستخراج التقرير
if st.sidebar.button("استخراج التقرير"):
    if "fire_detections" in st.session_state and st.session_state.fire_detections:
        filtered_detections = [
            detection for detection in st.session_state.fire_detections
            if start_date <= datetime.strptime(detection['time'], "%Y-%m-%d %H:%M:%S").date() <= end_date
        ]

        if filtered_detections:
            df = pd.DataFrame(filtered_detections)
            image_folder = "yolov5/runs/train/exp/images/"  # مسار مجلد الصور النسبي
            df['image_link'] = df['image'].apply(lambda x: f'=HYPERLINK("{image_folder}{x}", "عرض الصورة")')

            excel_file = "fire_detections_report.xlsx"
            df.to_excel(excel_file, index=False)

            with open(excel_file, "rb") as file:
                st.sidebar.download_button(
                    label="تحميل التقرير",
                    data=file,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.sidebar.error("لا توجد اكتشافات في الفترة المحددة.")
    else:
        st.sidebar.error("لا توجد اكتشافات لاستخراج التقرير.")

# باقي نظام اكتشاف الحرائق هنا
st.title("Fire Detection Monitoring System")
st.write(f"مرحباً بك في نظام اكتشاف الحريق")

# تحميل نموذج YOLOv5 فقط عند تشغيل الكشف
if "model" not in st.session_state:
    st.session_state.model = torch.hub.load('ultralytics/yolov5', 'custom', path='https://github.com/msj78598/Fire-Detection/raw/main/best.pt')

# إضافة مربع الإنذار الثابت في الأعلى
alert_box = st.empty()
alert_box.markdown("<div style='background-color: green; color: white; text-align: center; font-size: 24px;'>الوضع آمن ✔️</div>", unsafe_allow_html=True)

# عرض النظام الرئيسي
st.write("نظام مراقبة لاكتشاف الحريق")

# زر لبدء الفيديو
start_detection = st.button('ابدأ الكشف عن الحريق')

# متغيرات لتخزين بيانات الاكتشافات
if "fire_detections" not in st.session_state:
    st.session_state.fire_detections = []
if "fire_images" not in st.session_state:
    st.session_state.fire_images = []

# منطقة لعرض الفيديو
stframe = st.empty()  # مكان لعرض الفيديو الرئيسي
fire_images_placeholder = st.empty()  # منطقة لعرض الصور المكتشفة أسفل الفيديو

# فتح الكاميرا وتشغيل الكشف عند الضغط على زر البدء
if start_detection:
    cap = cv2.VideoCapture(0)

    fire_classes = [0, 1, 2, 3, 4]  # الفئات التي تمثل الحريق
    conf_threshold = 0.5  # تعديل عتبة الثقة إلى 50%

    # البدء في الحلقة
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("خطأ في فتح الكاميرا")
            break

        # استخدام النموذج للكشف
        results = st.session_state.model(frame)

        # فلترة الكائنات بناءً على عتبة الثقة
        detections = results.pandas().xyxy[0]
        detections = detections[detections['confidence'] > conf_threshold]  # 50% أو أكثر

        fire_detected = False
        for index, detection in detections.iterrows():
            if detection['class'] in fire_classes:
                fire_detected = True
                # الحصول على إحداثيات التحديد ونسبة الثقة
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                confidence = detection['confidence'] * 100  # نسبة الثقة مئوية

                # رسم مستطيل التحديد على منطقة الحريق
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # كتابة نسبة الثقة أعلى المستطيل
                cv2.putText(frame, f"Fire: {confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # الحصول على التاريخ والوقت
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                # إضافة التاريخ والوقت إلى الصورة
                cv2.putText(frame, f"Detected at: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # حفظ الصورة مع اسم يحتوي على التاريخ والوقت
                image_filename = f"fire_detected_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_filename, frame)

                # إضافة الصورة إلى قائمة الصور المكتشفة
                st.session_state.fire_images.insert(0, {
                    'image': image_filename,
                    'timestamp': timestamp
                })

                # تخزين بيانات الاكتشاف
                st.session_state.fire_detections.insert(0, {
                    'time': timestamp,
                    'image': image_filename,
                    'confidence': confidence
                })

                # تشغيل الإنذار اللوني مع وميض
                for _ in range(10):
                    alert_box.markdown("<div style='background-color: red; color: white; text-align: center; font-size: 24px;'>🚨🔥 إنذار حريق 🔥🚨</div>", unsafe_allow_html=True)
                    time.sleep(0.5)
                    alert_box.markdown("<div style='background-color: green; color: white; text-align: center; font-size: 24px;'>الوضع آمن ✔️</div>", unsafe_allow_html=True)
                    time.sleep(0.5)

        # تحويل الصورة إلى RGB لعرضها في Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # عرض الفيديو الرئيسي بحجم أكبر
        stframe.image(img_pil, width=700)  # تكبير حجم الفيديو الرئيسي

        # عرض الصور المكتشفة أسفل شاشة المراقبة مباشرة حسب الأحدث أولاً
        if st.session_state.fire_images:
            fire_images_placeholder.subheader("الصور المكتشفة")
            cols = fire_images_placeholder.columns(3)  # عرض الصور في ثلاثة أعمدة
            for idx, fire_image in enumerate(st.session_state.fire_images):
                cols[idx % 3].image(fire_image['image'], caption=f"اكتشاف في {fire_image['timestamp']}")

    cap.release()

'''
# !/usr/bin/env python3
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # oneDNN 연산 순서 고정
# import argparse, random, sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks, Input
# import matplotlib.pyplot as plt
# from PIL import Image, ImageFont, ImageDraw
# from tqdm import tqdm
# import albumentations as A

# # ====== 자모 리스트 ======
# CHO_LIST  = [chr(c) for c in [
#     0x3131,0x3132,0x3134,0x3137,0x3138,0x3139,
#     0x3141,0x3142,0x3143,0x3145,0x3146,0x3147,
#     0x3148,0x3149,0x314A,0x314B,0x314C,0x314D,0x314E
# ]]
# JUNG_LIST = [chr(c) for c in range(0x314F, 0x3164)]
# JONG_LIST = [''] + [chr(c) for c in [
#     0x3131,0x3132,0x3133,0x3134,0x3135,0x3136,
#     0x3137,0x3139,0x313A,0x313B,0x313C,0x313D,
#     0x313E,0x313F,0x3140,0x3141,0x3142,0x3144,
#     0x3145,0x3146,0x3147,0x3148,0x314A,0x314B,
#     0x314C,0x314D,0x314E
# ]]

# # ====== 인자 파싱 ======
# parser = argparse.ArgumentParser("Hangul Full Pipeline")
# parser.add_argument("--labels_txt",    type=Path, default=Path("labels/2350-common-hangul.txt"))
# parser.add_argument("--fonts_dir",     type=Path, default=Path("fonts"))
# parser.add_argument("--out_dir",       type=Path, default=Path("Images"))
# parser.add_argument("--size",          type=int,  default=64)
# parser.add_argument("--font_size",     type=int,  default=48)
# parser.add_argument("--augment",       type=int,  default=3)
# parser.add_argument("--val_split",     type=float,default=0.1)
# parser.add_argument("--seed",          type=int,  default=42)
# parser.add_argument("--batch",         type=int,  default=64)
# parser.add_argument("--epochs",        type=int,  default=20)
# parser.add_argument("--model_dir",     type=Path, default=Path("saved_model"))
# parser.add_argument("--predict_image", type=Path, help="예측할 단일 이미지")
# parser.add_argument("--predict_dir",   type=Path, help="예측할 이미지 폴더")
# parser.add_argument("--predict_csv",   type=Path, help="정확도 평가용 CSV")
# args = parser.parse_args()

# # ====== 1) 이미지 생성 (최초 1회) ======
# random.seed(args.seed)
# np.random.seed(args.seed)
# train_dir = args.out_dir/"train"
# valid_dir = args.out_dir/"valid"
# csv_path  = args.out_dir/"generated_labels.csv"

# if not csv_path.exists():
#     train_dir.mkdir(parents=True, exist_ok=True)
#     valid_dir.mkdir(parents=True, exist_ok=True)
#     labels = [l.strip() for l in args.labels_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
#     fonts  = sorted(args.fonts_dir.glob("*.ttf"))
#     if not fonts:
#         print(f"❌ .ttf 폰트 없음: {args.fonts_dir}"); sys.exit(1)

#     aug = A.Compose([
#         A.ElasticTransform(alpha=36, sigma=6, alpha_affine=3, p=0.5),
#         A.GaussNoise(p=0.2),
#         A.RandomBrightnessContrast(p=0.2)
#     ])

#     rows, img_id = [], 0
#     size_tuple = (args.size, args.size)

#     for ch in tqdm(labels, desc="Generating"):
#         code = ord(ch) - 0xAC00
#         ci, ji, oi = code//588, (code%588)//28, code%28
#         base = valid_dir if random.random() < args.val_split else train_dir
#         subp = base/ch
#         subp.mkdir(exist_ok=True)

#         for fp in fonts:
#             try:
#                 font = ImageFont.truetype(str(fp), args.font_size)
#             except:
#                 continue

#             img = Image.new("L", size_tuple, 0)
#             d   = ImageDraw.Draw(img)
#             bb  = d.textbbox((0,0), ch, font=font)
#             w,h = bb[2]-bb[0], bb[3]-bb[1]
#             d.text(((args.size-w)/2,(args.size-h)/2), ch, fill=255, font=font)

#             for k in range(args.augment+1):
#                 out_img = img if k==0 else Image.fromarray(aug(image=np.array(img))["image"])
#                 rgb     = out_img.convert("RGB")
#                 path    = subp/f"gen_{img_id:06d}.png"
#                 rgb.save(path)
#                 rows.append([str(path),ci,ji,oi,CHO_LIST[ci],JUNG_LIST[ji],JONG_LIST[oi]])
#                 img_id += 1

#     pd.DataFrame(rows, columns=[
#         "img_path","cho_idx","jung_idx","jong_idx",
#         "cho_char","jung_char","jong_char"
#     ]).to_csv(csv_path, index=False, encoding="utf-8")

#     print(f"✅ {len(rows)}개 이미지 생성 → {csv_path}")
# else:
#     print(f"✅ {csv_path} 존재 → 생성 스킵")

# # ====== 2) 학습 ======
# df = pd.read_csv(csv_path)

# if not df.img_path.str.contains("/train/").any():
#     df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
#     n  = int(len(df)*args.val_split)
#     df_train, df_val = df[n:], df[:n]
# else:
#     df_train = df[df.img_path.str.contains("/train/")]
#     df_val   = df[df.img_path.str.contains("/valid/")]

# def _map(p,c,j,o):
#     p   = tf.strings.regex_replace(p, "\\\\", "/")
#     img = tf.io.decode_png(tf.io.read_file(p), 3)
#     img = tf.image.resize(img,[args.size,args.size]) / 255.0
#     return img, {"cho":c,"jung":j,"jong":o}

# train_ds = tf.data.Dataset.from_tensor_slices((
#     df_train.img_path.values,
#     df_train.cho_idx.values,
#     df_train.jung_idx.values,
#     df_train.jong_idx.values
# )).map(_map, tf.data.AUTOTUNE).shuffle(1000).batch(args.batch).prefetch(tf.data.AUTOTUNE)

# val_ds = tf.data.Dataset.from_tensor_slices((
#     df_val.img_path.values,
#     df_val.cho_idx.values,
#     df_val.jung_idx.values,
#     df_val.jong_idx.values
# )).map(_map, tf.data.AUTOTUNE).batch(args.batch).prefetch(tf.data.AUTOTUNE)

# inp = Input((args.size,args.size,3)); x = inp
# for f in (32,64,128):
#     x = layers.Conv2D(f,3,activation="relu",padding="same")(x)
#     x = layers.MaxPooling2D()(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(256,activation="relu")(x)

# cho_out  = layers.Dense(19, activation="softmax", name="cho")(x)
# jung_out = layers.Dense(21, activation="softmax", name="jung")(x)
# jong_out = layers.Dense(28, activation="softmax", name="jong")(x)

# model = models.Model(inp, [cho_out, jung_out, jong_out])
# model.compile("adam",
#     loss={"cho":"sparse_categorical_crossentropy",
#           "jung":"sparse_categorical_crossentropy",
#           "jong":"sparse_categorical_crossentropy"},
#     metrics={"cho":"accuracy","jung":"accuracy","jong":"accuracy"}
# )
# model.summary()

# args.model_dir.mkdir(parents=True, exist_ok=True)
# ck = callbacks.ModelCheckpoint(args.model_dir/"best.keras",
#     monitor="val_loss", save_best_only=True, verbose=1)
# es = callbacks.EarlyStopping(monitor="val_loss", patience=5,
#     restore_best_weights=True, verbose=1)

# hist = model.fit(train_ds, validation_data=val_ds,
#     epochs=args.epochs, callbacks=[ck, es])

# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(hist.history["loss"],label="train_loss")
# plt.plot(hist.history["val_loss"],label="val_loss")
# plt.title("Loss"); plt.legend()

# plt.subplot(1,2,2)
# plt.plot(hist.history["cho_accuracy"],label="cho_acc")
# plt.plot(hist.history["val_cho_accuracy"],label="val_cho_acc")
# plt.title("초성 Accuracy"); plt.legend()

# plt.tight_layout(); plt.show()
# print(f"✅ 모델 저장 → {args.model_dir/'best.keras'}")

# # ====== 3) 예측/평가 ======
# model = tf.keras.models.load_model(str(args.model_dir/"best.keras"))

# if args.predict_image:
#     p = args.predict_image
#     if not p.exists(): print(f"❌ 이미지 없음: {p}"); sys.exit(1)
#     img = Image.open(p).convert("RGB").resize((args.size,args.size))
#     X = (np.array(img,np.float32)/255.0)[np.newaxis,...]
#     ch,jg,jo = model.predict(X,verbose=0)
#     c,j,o = int(np.argmax(ch)),int(np.argmax(jg)),int(np.argmax(jo))
#     print(f"예측: {chr(0xAC00+c*588+j*28+o)} (초{CHO_LIST[c]}, 중{JUNG_LIST[j]}, 종{JONG_LIST[o]})")

# elif args.predict_dir:
#     d = args.predict_dir
#     if not d.is_dir(): print(f"❌ 폴더 없음: {d}"); sys.exit(1)
#     imgs = sorted(d.glob("*.png"))
#     print(f"🔍 {len(imgs)}개 예측 시작...")
#     for i in range(0, len(imgs), args.batch):
#         blk, arrs = imgs[i:i+args.batch], []
#         for p in blk:
#             img = Image.open(p).convert("RGB").resize((args.size,args.size))
#             arrs.append(np.array(img,np.float32)/255.0)
#         X = np.stack(arrs,0)
#         ch,jg,jo = model.predict(X,verbose=0)
#         ci,ji,oi = np.argmax(ch,1),np.argmax(jg,1),np.argmax(jo,1)
#         for p,c,j,o in zip(blk,ci,ji,oi):
#             print(f"{p.name} → {chr(0xAC00+c*588+j*28+o)} (초{CHO_LIST[c]}, 중{JUNG_LIST[j]}, 종{JONG_LIST[o]})")

# else:
#     eval_csv = args.predict_csv or csv_path
#     df_eval  = pd.read_csv(eval_csv)
#     valid    = df_eval[df_eval.img_path.str.contains("valid", na=False)]
#     if valid.empty: print("❌ valid 샘플 없음"); sys.exit(1)
#     paths = [Path(p) for p in valid.img_path]
#     true  = valid[["cho_idx","jung_idx","jong_idx"]].values.astype(int)
#     preds = []
#     for i in range(0, len(paths), args.batch):
#         blk, arrs = paths[i:i+args.batch], []
#         for p in blk:
#             img = Image.open(p).convert("RGB").resize((args.size,args.size))
#             arrs.append(np.array(img,np.float32)/255.0)
#         X = np.stack(arrs,0)
#         ch,jg,jo = model.predict(X,verbose=0)
#         ci,ji,oi = np.argmax(ch,1),np.argmax(jg,1),np.argmax(jo,1)
#         preds.extend(zip(ci,ji,oi))
#     preds   = np.array(preds)
#     cho_acc  = (preds[:,0]==true[:,0]).mean()
#     jung_acc = (preds[:,1]==true[:,1]).mean()
#     jong_acc = (preds[:,2]==true[:,2]).mean()
#     full_acc = np.all(preds==true,axis=1).mean()
#     print(f"▶ 초성 {cho_acc*100:.2f}%, 중성 {jung_acc*100:.2f}%, 종성 {jong_acc*100:.2f}%, 완전일치 {full_acc*100:.2f}%")
'''

'''
#!/usr/bin/env python3
# ====== 설정 ======
# LABELS_TXT    = "labels/2350-common-hangul.txt" # 라벨 파일 경로
# FONTS_DIR     = "fonts"                         # TTF 폰트 폴더
# OUT_DIR       = "Images"                        # 이미지 및 CSV 저장 폴더
# SIZE          = 64                              # 출력 이미지 한 변 크기
# FONT_SIZE     = 48                              # 렌더링 폰트 크기
# AUGMENT_CNT   = 3                               # 원본당 증강 이미지 수
# VAL_SPLIT     = 0.1                             # 검증 세트 비율
# SEED          = 42                              # 난수 시드
# BATCH_SIZE    = 64                              # 학습 배치 크기
# EPOCHS        = 20                              # 학습 최대 Epoch 수
# MODEL_DIR     = "saved_model"                   # 모델 저장 폴더
# PREDICT_IMAGE = None                            # 단일 이미지 예측 (예: "Images/valid/...")
# PREDICT_DIR   = None                            # 폴더 예측 (예: "Images/valid")
# PREDICT_CSV   = None                            # CSV 정확도 평가 (예: "Images/generated_labels.csv")

# # ====== 라이브러리 임포트 ======
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # oneDNN 연산 순서 고정
# import sys, random, numpy as np, pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks, Input
# import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image, ImageFont, ImageDraw
# from tqdm import tqdm
# import albumentations as A

# # ====== 자모 리스트 ======
# CHO_LIST  = [chr(c) for c in [
#     0x3131,0x3132,0x3134,0x3137,0x3138,0x3139,
#     0x3141,0x3142,0x3143,0x3145,0x3146,0x3147,
#     0x3148,0x3149,0x314A,0x314B,0x314C,0x314D,0x314E
# ]]
# JUNG_LIST = [chr(c) for c in range(0x314F, 0x3164)]
# JONG_LIST = [''] + [chr(c) for c in [
#     0x3131,0x3132,0x3133,0x3134,0x3135,0x3136,
#     0x3137,0x3139,0x313A,0x313B,0x313C,0x313D,
#     0x313E,0x313F,0x3140,0x3141,0x3142,0x3144,
#     0x3145,0x3146,0x3147,0x3148,0x314A,0x314B,
#     0x314C,0x314D,0x314E
# ]]

# # ====== 1) 이미지 생성 & CSV (최초 1회) ======
# random.seed(SEED)
# np.random.seed(SEED)

# out_dir   = Path(OUT_DIR)
# train_dir = out_dir/"train"
# valid_dir = out_dir/"valid"
# csv_path  = out_dir/"generated_labels.csv"

# if not csv_path.exists():
#     train_dir.mkdir(parents=True, exist_ok=True)
#     valid_dir.mkdir(parents=True, exist_ok=True)

#     labels = [
#         l.strip() for l in Path(LABELS_TXT).read_text(encoding="utf-8").splitlines()
#         if l.strip()
#     ]
#     fonts = sorted(Path(FONTS_DIR).glob("*.ttf"))
#     if not fonts:
#         print(f"❌ .ttf 폰트가 없습니다: {FONTS_DIR}")
#         sys.exit(1)

#     aug = A.Compose([
#         A.ElasticTransform(alpha=36, sigma=6, alpha_affine=3, p=0.5),
#         A.GaussNoise(p=0.2),
#         A.RandomBrightnessContrast(p=0.2)
#     ])

#     rows, img_id = [], 0
#     size_tuple = (SIZE, SIZE)

#     for ch in tqdm(labels, desc="Generating"):
#         code = ord(ch) - 0xAC00
#         ci, ji, oi = code//588, (code%588)//28, code%28
#         base = valid_dir if random.random() < VAL_SPLIT else train_dir
#         subp = base / ch
#         subp.mkdir(exist_ok=True)

#         for fp in fonts:
#             try:
#                 font = ImageFont.truetype(str(fp), FONT_SIZE)
#             except:
#                 continue

#             img = Image.new("L", size_tuple, 0)
#             d   = ImageDraw.Draw(img)
#             bb  = d.textbbox((0,0), ch, font=font)
#             w,h = bb[2]-bb[0], bb[3]-bb[1]
#             d.text(((SIZE-w)/2, (SIZE-h)/2), ch, fill=255, font=font)

#             for k in range(AUGMENT_CNT+1):
#                 out_img = img if k==0 else Image.fromarray(
#                     aug(image=np.array(img))["image"]
#                 )
#                 rgb  = out_img.convert("RGB")
#                 path = subp/f"gen_{img_id:06d}.png"
#                 rgb.save(path)
#                 rows.append([
#                     str(path),
#                     ci, ji, oi,
#                     CHO_LIST[ci], JUNG_LIST[ji], JONG_LIST[oi]
#                 ])
#                 img_id += 1

#     pd.DataFrame(rows, columns=[
#         "img_path","cho_idx","jung_idx","jong_idx",
#         "cho_char","jung_char","jong_char"
#     ]).to_csv(csv_path, index=False, encoding="utf-8")
#     print(f"✅ {len(rows)}개 이미지 생성 → {csv_path}")
# else:
#     print(f"✅ {csv_path} 존재 → 생성 스킵")

# # ====== 2) 학습 ======
# df = pd.read_csv(csv_path)

# if not df.img_path.str.contains("/train/").any():
#     df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
#     n  = int(len(df) * VAL_SPLIT)
#     df_train, df_val = df[n:], df[:n]
# else:
#     df_train = df[df.img_path.str.contains("/train/")]
#     df_val   = df[df.img_path.str.contains("/valid/")]

# def _map(path, c, j, o):
#     p   = tf.strings.regex_replace(path, "\\\\", "/")
#     img = tf.image.decode_png(tf.io.read_file(p), channels=3)
#     img = tf.image.resize(img, [SIZE, SIZE]) / 255.0
#     return img, {"cho": c, "jung": j, "jong": o}

# train_ds = tf.data.Dataset.from_tensor_slices((
#     df_train.img_path.values,
#     df_train.cho_idx.values,
#     df_train.jung_idx.values,
#     df_train.jong_idx.values
# )).map(_map, tf.data.AUTOTUNE).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# val_ds = tf.data.Dataset.from_tensor_slices((
#     df_val.img_path.values,
#     df_val.cho_idx.values,
#     df_val.jung_idx.values,
#     df_val.jong_idx.values
# )).map(_map, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# inp = Input((SIZE, SIZE, 3)); x = inp
# for f in (32, 64, 128):
#     x = layers.Conv2D(f, 3, activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D()(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(256, activation="relu")(x)

# cho_out  = layers.Dense(19, activation="softmax", name="cho")(x)
# jung_out = layers.Dense(21, activation="softmax", name="jung")(x)
# jong_out = layers.Dense(28, activation="softmax", name="jong")(x)

# model = models.Model(inp, [cho_out, jung_out, jong_out])
# model.compile(
#     "adam",
#     loss={"cho":"sparse_categorical_crossentropy",
#           "jung":"sparse_categorical_crossentropy",
#           "jong":"sparse_categorical_crossentropy"},
#     metrics={"cho":"accuracy","jung":"accuracy","jong":"accuracy"}
# )
# model.summary()

# Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
# ck = callbacks.ModelCheckpoint(
#     Path(MODEL_DIR)/"best.keras",
#     monitor="val_loss", save_best_only=True, verbose=1
# )
# es = callbacks.EarlyStopping(
#     monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
# )

# hist = model.fit(
#     train_ds, validation_data=val_ds,
#     epochs=EPOCHS, callbacks=[ck, es]
# )

# # ====== 학습 결과 시각화 ======
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.plot(hist.history["loss"],     label="train_loss")
# plt.plot(hist.history["val_loss"], label="val_loss")
# plt.title("Loss"); plt.legend()

# plt.subplot(1,2,2)
# plt.plot(hist.history["cho_accuracy"],     label="cho_acc")
# plt.plot(hist.history["val_cho_accuracy"], label="val_cho_acc")
# plt.title("초성 Accuracy"); plt.legend()

# plt.tight_layout(); plt.show()
# print(f"✅ 모델 저장 → {MODEL_DIR}/best.keras")

# # ====== 3) 예측/평가 ======
# model = tf.keras.models.load_model(str(Path(MODEL_DIR)/"best.keras"))

# if PREDICT_IMAGE:
#     p = Path(PREDICT_IMAGE)
#     if not p.exists(): print(f"❌ 이미지 없음: {p}"); sys.exit(1)
#     img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
#     X = (np.array(img, np.float32)/255.0)[None,...]
#     ch, jg, jo = model.predict(X, verbose=0)
#     c,j,o = map(int, (ch.argmax(), jg.argmax(), jo.argmax()))
#     print(f"예측: {chr(0xAC00+c*588+j*28+o)} " +
#           f"(초{CHO_LIST[c]}, 중{JUNG_LIST[j]}, 종{JONG_LIST[o]})")

# elif PREDICT_DIR:
#     d = Path(PREDICT_DIR)
#     if not d.is_dir(): print(f"❌ 폴더 없음: {d}"); sys.exit(1)
#     imgs = sorted(d.glob("*.png"))
#     print(f"🔍 {len(imgs)}개 예측 시작...")
#     for i in range(0, len(imgs), BATCH_SIZE):
#         blk, arrs = imgs[i:i+BATCH_SIZE], []
#         for p in blk:
#             img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
#             arrs.append(np.array(img, np.float32)/255.0)
#         X = np.stack(arrs, 0)
#         ch, jg, jo = model.predict(X, verbose=0)
#         ci, ji, oi = ch.argmax(1), jg.argmax(1), jo.argmax(1)
#         for p,c,j,o in zip(blk,ci,ji,oi):
#             print(f"{p.name} → {chr(0xAC00+c*588+j*28+o)} " +
#                   f"(초{CHO_LIST[c]}, 중{JUNG_LIST[j]}, 종{JONG_LIST[o]})")

# else:
#     eval_csv = PREDICT_CSV or str(csv_path)
#     df_eval  = pd.read_csv(eval_csv)
#     valid    = df_eval[df_eval.img_path.str.contains("valid", na=False)]
#     if valid.empty: print("❌ valid 샘플 없음"); sys.exit(1)
#     paths = [Path(p) for p in valid.img_path]
#     true  = valid[["cho_idx","jung_idx","jong_idx"]].values.astype(int)
#     preds = []
#     for i in range(0, len(paths), BATCH_SIZE):
#         blk, arrs = paths[i:i+BATCH_SIZE], []
#         for p in blk:
#             img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
#             arrs.append(np.array(img, np.float32)/255.0)
#         X = np.stack(arrs, 0)
#         ch, jg, jo = model.predict(X, verbose=0)
#         ci, ji, oi = ch.argmax(1), jg.argmax(1), jo.argmax(1)
#         preds.extend(zip(ci, ji, oi))
#     preds    = np.array(preds)
#     cho_acc  = (preds[:,0] == true[:,0]).mean()
#     jung_acc = (preds[:,1] == true[:,1]).mean()
#     jong_acc = (preds[:,2] == true[:,2]).mean()
#     full_acc = np.all(preds == true, axis=1).mean()
#     print(f"▶ 초성 {cho_acc*100:.2f}%, 중성 {jung_acc*100:.2f}%, 종성 {jong_acc*100:.2f}%, 완전일치 {full_acc*100:.2f}%")
'''

'''
#!/usr/bin/env python3
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # oneDNN 연산 순서 고정

# import random
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks, Input
# import matplotlib.pyplot as plt

# from PIL import Image, ImageFont, ImageDraw
# from tqdm import tqdm
# import albumentations as A

# plt.rcParams['font.family'] = 'Malgun Gothic'

# # ====== Settings ======
# LABELS_TXT  = Path("labels/2350-common-hangul.txt")
# FONTS_DIR   = Path("fonts")
# OUT_DIR     = Path("Images")
# SIZE        = 64
# FONT_SIZE   = 48
# AUGMENT_CNT = 3
# VAL_SPLIT   = 0.1
# SEED        = 42
# BATCH_SIZE  = 64
# EPOCHS      = 20
# MODEL_DIR   = Path("saved_model")
# MODEL_FILE  = MODEL_DIR / "best.keras"
# CSV_PATH    = OUT_DIR / "generated_labels.csv"

# # ====== Jamo lists ======
# CHO_LIST  = [chr(c) for c in [
#     0x3131,0x3132,0x3134,0x3137,0x3138,0x3139,
#     0x3141,0x3142,0x3143,0x3145,0x3146,0x3147,
#     0x3148,0x3149,0x314A,0x314B,0x314C,0x314D,0x314E
# ]]
# JUNG_LIST = [chr(c) for c in range(0x314F, 0x3164)]
# JONG_LIST = [''] + [chr(c) for c in [
#     0x3131,0x3132,0x3133,0x3134,0x3135,0x3136,
#     0x3137,0x3139,0x313A,0x313B,0x313C,0x313D,
#     0x313E,0x313F,0x3140,0x3141,0x3142,0x3144,
#     0x3145,0x3146,0x3147,0x3148,0x314A,0x314B,
#     0x314C,0x314D,0x314E
# ]]

# # ====== 1) Image generation & CSV (run once) ======
# random.seed(SEED)
# np.random.seed(SEED)

# train_dir = OUT_DIR / "train"
# valid_dir = OUT_DIR / "valid"

# if not CSV_PATH.exists():
#     train_dir.mkdir(parents=True, exist_ok=True)
#     valid_dir.mkdir(parents=True, exist_ok=True)

#     labels = [l.strip() for l in LABELS_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
#     fonts  = sorted(FONTS_DIR.glob("*.ttf"))
#     if not fonts:
#         print(f"❌ No .ttf fonts in {FONTS_DIR}")
#         sys.exit(1)

#     aug = A.Compose([
#         A.ElasticTransform(alpha=36, sigma=6, alpha_affine=3, p=0.5),
#         A.GaussNoise(p=0.2),
#         A.RandomBrightnessContrast(p=0.2)
#     ])

#     rows = []
#     img_id = 0
#     size_tuple = (SIZE, SIZE)

#     for ch in tqdm(labels, desc="Generating"):
#         code = ord(ch) - 0xAC00
#         ci, ji, oi = code // 588, (code % 588) // 28, code % 28
#         base = valid_dir if random.random() < VAL_SPLIT else train_dir
#         subp = base / ch
#         subp.mkdir(exist_ok=True)

#         for fp in fonts:
#             try:
#                 font = ImageFont.truetype(str(fp), FONT_SIZE)
#             except:
#                 continue

#             img  = Image.new("L", size_tuple, 0)
#             draw = ImageDraw.Draw(img)
#             bb   = draw.textbbox((0,0), ch, font=font)
#             w, h = bb[2] - bb[0], bb[3] - bb[1]
#             draw.text(((SIZE-w)/2, (SIZE-h)/2), ch, fill=255, font=font)

#             for k in range(AUGMENT_CNT + 1):
#                 if k == 0:
#                     out_img = img
#                 else:
#                     arr     = np.array(img).astype(np.uint8)
#                     out_img = Image.fromarray(aug(image=arr)["image"])
#                 rgb  = out_img.convert("RGB")
#                 path = subp / f"gen_{img_id:06d}.png"
#                 rgb.save(path)
#                 rows.append([
#                     str(path),
#                     ci, ji, oi,
#                     CHO_LIST[ci], JUNG_LIST[ji], JONG_LIST[oi]
#                 ])
#                 img_id += 1

#     pd.DataFrame(rows, columns=[
#         "img_path","cho_idx","jung_idx","jong_idx",
#         "cho_char","jung_char","jong_char"
#     ]).to_csv(CSV_PATH, index=False, encoding="utf-8")
#     print(f"✅ Generated {len(rows)} images → {CSV_PATH}")
# else:
#     print(f"✅ {CSV_PATH} exists → skipping image generation")

# # ====== 2) Read & split ======
# df = pd.read_csv(CSV_PATH)
# df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
# n  = int(len(df) * VAL_SPLIT)
# df_val, df_train = df[:n], df[n:]

# # ====== 3) Build tf.data datasets ======
# def _map(path, c, j, o):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, [SIZE, SIZE]) / 255.0
#     return img, {"cho": c, "jung": j, "jong": o}

# train_ds = tf.data.Dataset.from_tensor_slices((
#     df_train.img_path.values,
#     df_train.cho_idx.values,
#     df_train.jung_idx.values,
#     df_train.jong_idx.values
# )).map(_map, num_parallel_calls=tf.data.AUTOTUNE) \
#   .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# val_ds = tf.data.Dataset.from_tensor_slices((
#     df_val.img_path.values,
#     df_val.cho_idx.values,
#     df_val.jung_idx.values,
#     df_val.jong_idx.values
# )).map(_map, num_parallel_calls=tf.data.AUTOTUNE) \
#   .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# # ====== 4) Build or load model ======
# if not MODEL_FILE.exists():
#     inp = Input((SIZE, SIZE, 3))
#     x   = inp
#     for f in (32, 64, 128):
#         x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
#         x = layers.MaxPooling2D()(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(256, activation="relu")(x)

#     cho_out  = layers.Dense(19, activation="softmax", name="cho")(x)
#     jung_out = layers.Dense(21, activation="softmax", name="jung")(x)
#     jong_out = layers.Dense(28, activation="softmax", name="jong")(x)

#     model = models.Model(inp, [cho_out, jung_out, jong_out])
#     model.compile(
#         optimizer="adam",
#         loss={
#             "cho":  "sparse_categorical_crossentropy",
#             "jung": "sparse_categorical_crossentropy",
#             "jong": "sparse_categorical_crossentropy"
#         },
#         metrics={
#             "cho":  "accuracy",
#             "jung": "accuracy",
#             "jong": "accuracy"
#         }
#     )

#     MODEL_DIR.mkdir(parents=True, exist_ok=True)
#     ck = callbacks.ModelCheckpoint(
#         MODEL_FILE, monitor="val_loss", save_best_only=True, verbose=1
#     )
#     es = callbacks.EarlyStopping(
#         monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
#     )

#     hist = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=EPOCHS,
#         callbacks=[ck, es]
#     )

#     # Training curves
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,2,1)
#     plt.plot(hist.history["loss"],      label="train_loss")
#     plt.plot(hist.history["val_loss"],  label="val_loss")
#     plt.title("Loss"); plt.legend()

#     plt.subplot(1,2,2)
#     plt.plot(hist.history["cho_accuracy"],     label="cho_acc")
#     plt.plot(hist.history["val_cho_accuracy"], label="val_cho_acc")
#     plt.title("초성 Accuracy"); plt.legend()

#     plt.tight_layout()
#     plt.show()
#     print(f"✅ Training complete → {MODEL_FILE}")
# else:
#     print(f"✅ Loading existing model → {MODEL_FILE}")
#     model = tf.keras.models.load_model(str(MODEL_FILE))

# # ====== 5) Preview a few validation predictions ======
# for imgs, labels in val_ds.take(1):
#     preds = model.predict(imgs, verbose=0)
#     ci = np.argmax(preds[0], axis=1)
#     ji = np.argmax(preds[1], axis=1)
#     oi = np.argmax(preds[2], axis=1)
#     cho_t  = labels["cho"].numpy()
#     jung_t = labels["jung"].numpy()
#     jong_t = labels["jong"].numpy()

#     n = min(9, imgs.shape[0])
#     plt.figure(figsize=(7,7))
#     for i in range(n):
#         ax = plt.subplot(3,3,i+1)
#         ax.imshow((imgs[i].numpy()*255).astype("uint8"))
#         true_c = chr(0xAC00 + cho_t[i]*588 + jung_t[i]*28 + jong_t[i])
#         pred_c = chr(0xAC00 + ci[i]*588      + ji[i]*28      + oi[i])
#         ax.set_title(f"실제:{true_c}\n예측:{pred_c}", fontsize=10)
#         ax.axis("off")
#     plt.suptitle("Validation Sample Predictions", fontsize=14)
#     plt.tight_layout()
#     plt.show()
#     break

# # ====== 6) Full validation accuracy ======
# true = df_val[["cho_idx","jung_idx","jong_idx"]].values
# preds = []
# paths = df_val.img_path.values
# for i in range(0, len(paths), BATCH_SIZE):
#     blk = paths[i:i+BATCH_SIZE]
#     arrs = []
#     for p in blk:
#         img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
#         arrs.append(np.array(img, np.float32)/255.0)
#     X = np.stack(arrs, 0)
#     ch, jg, jo = model.predict(X, verbose=0)
#     ci, ji, oi = ch.argmax(1), jg.argmax(1), jo.argmax(1)
#     preds.extend(zip(ci, ji, oi))

# preds = np.array(preds, dtype=int)

# cho_acc  = (preds[:,0] == true[:,0]).mean()
# jung_acc = (preds[:,1] == true[:,1]).mean()
# jong_acc = (preds[:,2] == true[:,2]).mean()
# full_acc = np.all(preds == true, axis=1).mean()

# print(f"▶ 초성 {cho_acc*100:.2f}%, 중성 {jung_acc*100:.2f}%, 종성 {jong_acc*100:.2f}%, 완전일치 {full_acc*100:.2f}%")
'''

#!/usr/bin/env python3
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # oneDNN 연산 순서 고정

import random
import sys
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import albumentations as A

plt.rcParams['font.family'] = 'Malgun Gothic'

# ====== 설정 ======
LABELS_TXT  = Path("labels/2350-common-hangul.txt")
FONTS_DIR   = Path("fonts")
OUT_DIR     = Path("Images")
SIZE        = 64
FONT_SIZE   = 48
AUGMENT_CNT = 3
VAL_SPLIT   = 0.1
SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 20
MODEL_DIR   = Path("saved_model")
MODEL_FILE  = MODEL_DIR / "best.keras"
CSV_PATH    = OUT_DIR / "generated_labels.csv"

# ====== 자모 리스트 ======
CHO_LIST  = [chr(c) for c in [
    0x3131,0x3132,0x3134,0x3137,0x3138,0x3139,
    0x3141,0x3142,0x3143,0x3145,0x3146,0x3147,
    0x3148,0x3149,0x314A,0x314B,0x314C,0x314D,0x314E
]]
JUNG_LIST = [chr(c) for c in range(0x314F, 0x3164)]
JONG_LIST = [''] + [chr(c) for c in [
    0x3131,0x3132,0x3133,0x3134,0x3135,0x3136,
    0x3137,0x3139,0x313A,0x313B,0x313C,0x313D,
    0x313E,0x313F,0x3140,0x3141,0x3142,0x3144,
    0x3145,0x3146,0x3147,0x3148,0x314A,0x314B,
    0x314C,0x314D,0x314E
]]

# ====== 1) 이미지 생성 및 CSV 저장 (최초 1회) ======
random.seed(SEED)
np.random.seed(SEED)

train_dir = OUT_DIR / "train"
valid_dir = OUT_DIR / "valid"

if not CSV_PATH.exists():
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    labels = [l.strip() for l in LABELS_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
    fonts  = sorted(FONTS_DIR.glob("*.ttf"))
    if not fonts:
        print(f"❌ {FONTS_DIR}에 .ttf 폰트가 없습니다.")
        sys.exit(1)

    aug = A.Compose([
        A.ElasticTransform(alpha=36, sigma=6, alpha_affine=3, p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2)
    ])

    rows = []
    img_id = 0
    for ch in tqdm(labels, desc="이미지 생성 중"):
        code = ord(ch) - 0xAC00
        ci, ji, oi = code // 588, (code % 588) // 28, code % 28
        base = valid_dir if random.random() < VAL_SPLIT else train_dir
        subp = base / ch
        subp.mkdir(exist_ok=True)

        for fp in fonts:
            try:
                font = ImageFont.truetype(str(fp), FONT_SIZE)
            except:
                continue

            img  = Image.new("L", (SIZE, SIZE), 0)
            draw = ImageDraw.Draw(img)
            bb   = draw.textbbox((0,0), ch, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            draw.text(((SIZE-w)/2, (SIZE-h)/2), ch, fill=255, font=font)

            for k in range(AUGMENT_CNT + 1):
                if k == 0:
                    out_img = img
                else:
                    arr     = np.array(img).astype(np.uint8)
                    out_img = Image.fromarray(aug(image=arr)["image"])
                rgb  = out_img.convert("RGB")
                path = subp / f"gen_{img_id:06d}.png"
                rgb.save(path)
                rows.append([
                    str(path),
                    ci, ji, oi,
                    CHO_LIST[ci], JUNG_LIST[ji], JONG_LIST[oi]
                ])
                img_id += 1

    pd.DataFrame(rows, columns=[
        "img_path","cho_idx","jung_idx","jong_idx",
        "cho_char","jung_char","jong_char"
    ]).to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"✅ {len(rows)}개 이미지 생성 완료 → {CSV_PATH}")
else:
    print(f"✅ {CSV_PATH}이(가) 이미 존재합니다. 이미지 생성 생략")

# ====== 2) 데이터 읽기 및 train/validation 분할 ======
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n  = int(len(df) * VAL_SPLIT)
df_val, df_train = df[:n], df[n:]

# ====== 3) tf.data.Dataset 생성 ======
def _map(path, c, j, o):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [SIZE, SIZE]) / 255.0
    return img, {"cho": c, "jung": j, "jong": o}

train_ds = tf.data.Dataset.from_tensor_slices((
    df_train.img_path.values,
    df_train.cho_idx.values,
    df_train.jung_idx.values,
    df_train.jong_idx.values
)).map(_map, num_parallel_calls=tf.data.AUTOTUNE) \
  .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((
    df_val.img_path.values,
    df_val.cho_idx.values,
    df_val.jung_idx.values,
    df_val.jong_idx.values
)).map(_map, num_parallel_calls=tf.data.AUTOTUNE) \
  .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ====== 4) 모델 생성 또는 로드 및 학습 ======
if not MODEL_FILE.exists():
    inp = Input((SIZE, SIZE, 3))
    x   = inp
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    cho_out  = layers.Dense(19, activation="softmax", name="cho")(x)
    jung_out = layers.Dense(21, activation="softmax", name="jung")(x)
    jong_out = layers.Dense(28, activation="softmax", name="jong")(x)

    model = models.Model(inp, [cho_out, jung_out, jong_out])
    model.compile(
        optimizer="adam",
        loss={
            "cho":  "sparse_categorical_crossentropy",
            "jung": "sparse_categorical_crossentropy",
            "jong": "sparse_categorical_crossentropy"
        },
        metrics={
            "cho":  "accuracy",
            "jung": "accuracy",
            "jong": "accuracy"
        }
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ck = callbacks.ModelCheckpoint(
        MODEL_FILE, monitor="val_loss", save_best_only=True, verbose=1
    )
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ck, es]
    )

    # 학습 곡선 시각화
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history["loss"],      label="train_loss")
    plt.plot(hist.history["val_loss"],  label="val_loss")
    plt.title("손실(Loss)"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history["cho_accuracy"],     label="cho_acc")
    plt.plot(hist.history["val_cho_accuracy"], label="val_cho_acc")
    plt.title("초성 정확도"); plt.legend()

    plt.tight_layout()
    plt.show()
    print(f"✅ 학습 완료 → {MODEL_FILE}")
else:
    print(f"✅ 기존 모델 로드 → {MODEL_FILE}")
    model = tf.keras.models.load_model(str(MODEL_FILE))

# ====== 5) 랜덤 검증 배치 샘플 예측 미리보기 ======
num_val_batches = math.ceil(len(df_val) / BATCH_SIZE)
random_val_ds = val_ds.shuffle(
    buffer_size=num_val_batches,
    seed=int(time.time())
)
for imgs, labels in random_val_ds.take(1):
    preds = model.predict(imgs, verbose=0)
    ci    = np.argmax(preds[0], axis=1)
    ji    = np.argmax(preds[1], axis=1)
    oi    = np.argmax(preds[2], axis=1)
    cho_t  = labels["cho"].numpy()
    jung_t = labels["jung"].numpy()
    jong_t = labels["jong"].numpy()

    n = min(25, imgs.shape[0])  # 최대 5*5 = 25장
    plt.figure(figsize=(8,8))
    for i in range(n):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow((imgs[i].numpy() * 255).astype("uint8"))
        실제 = chr(0xAC00 + cho_t[i] * 588 + jung_t[i] * 28 + jong_t[i])
        예측 = chr(0xAC00 + ci[i]     * 588 + ji[i]     * 28 + oi[i])
        ax.set_title(f"실제:{실제}\n예측:{예측}", fontsize=8)
        ax.axis("off")
    plt.suptitle("검증 샘플 예측 결과 (5x5)", fontsize=16)
    plt.tight_layout()
    plt.show()
    break

# ====== 6) 전체 검증 정확도 평가 ======
true = df_val[["cho_idx","jung_idx","jong_idx"]].values
preds = []
paths = df_val.img_path.values
for i in range(0, len(paths), BATCH_SIZE):
    blk  = paths[i:i+BATCH_SIZE]
    arrs = []
    for p in blk:
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
        arrs.append(np.array(img, np.float32)/255.0)
    X       = np.stack(arrs, 0)
    ch, jg, jo = model.predict(X, verbose=0)
    ci, ji, oi = ch.argmax(1), jg.argmax(1), jo.argmax(1)
    preds.extend(zip(ci, ji, oi))

preds = np.array(preds, dtype=int)

cho_acc  = (preds[:,0] == true[:,0]).mean()
jung_acc = (preds[:,1] == true[:,1]).mean()
jong_acc = (preds[:,2] == true[:,2]).mean()
full_acc = np.all(preds == true, axis=1).mean()

print(f"▶ 초성 정확도: {cho_acc*100:.2f}%")
print(f"▶ 중성 정확도: {jung_acc*100:.2f}%")
print(f"▶ 종성 정확도: {jong_acc*100:.2f}%")
print(f"▶ 전체 문자 일치율: {full_acc*100:.2f}%")

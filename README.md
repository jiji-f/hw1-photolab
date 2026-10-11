# Запуск фотолаборатории

Программа и фотографии находятся в этой папке. Откройте терминал в корне этого репозитория и установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

Для воспроизведения результатов:

```powershell
python photolab.py gamma data/dark.jpeg --gamma 0.5 -o out/dark_g05.jpg --hist out/dark_g05_hist.png
python photolab.py gamma data/dark.jpeg --gamma 2 -o out/dark_g2.jpg --hist out/dark_g2_hist.png
python photolab.py gamma data/bright.jpeg --gamma 2 -o out/bright_g2.jpg --hist out/bright_g2_hist.png
python photolab.py wb data/bright.jpeg --method grey-world -o out/bright_gw.jpg
python photolab.py wb data/bright.jpeg --method white-patch -o out/bright_wp.jpg
python photolab.py clahe data/dark.jpeg --clip 2 --tiles 8 -o out/dark_clahe.jpg --compare out/dark_cmp.jpg --hist out/dark_clahe_hist.png
python photolab.py rotate data/poster.jpeg --deg 30 -o out/poster_rot30.jpg
python photolab.py resize data/text.jpeg --scale 0.25 --report -o out/text_x025.jpg
python photolab.py rectify data/poster.jpeg --corners 226.74285888671875,137.14285278320312,833.8285522460938,354.74285888671875,874.05712890625,1102.6285400390625,120.68571472167969,1210.5142822265625 --aspect 0.707 -o out/poster_front.jpg
python orange_object.py
python photolab.py sheet data/object.jpeg out/object_on_bg.jpg out/object_mask.png out/poster_front.jpg --cols 2 -o out/sheet.jpg
python check_samples.py
python -m pytest tests -q
```

Числа и пояснения: [REPORT.md](REPORT.md). `orange_object.py` воспроизводит выделение лицевой стороны оранжевой коробки. `check_samples.py` требует папку `lectures` из репозитория курса и не создаёт файлов в `out`. Проверка: `python -m pytest tests -q`.

Шаблон CI лежит в `.github/workflows`. Если эту папку копируют в отдельный репозиторий Classroom, её содержимое должно находиться в корне этого репозитория вместе с `photolab.py`, `requirements.txt` и `tests`.


`check_samples.py` для отдельного репозитория: `python check_samples.py --lectures D:/CV/iu-intro-cv/lectures` (подставьте свой путь к курсу).

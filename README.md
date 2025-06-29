סווג תמונות עם רשת נוירונים

🎯 מטרת הפרויקט

הפרויקט עוסק בסיווג תמונות של **בניינים לעומת יערות** באמצעות רשתות נוירונים קונבולוציוניות (CNN), תוך שימוש בגישות של אימון מאפס וגם Fine-Tuning על מודל קיים (ResNet18).

🛠 טכנולוגיות וכלים

Python

PyTorch

CNN פשוטה

ResNet18 (Fine-Tuning)

Kaggle Dataset: Buildings vs Forests

Augmentation, ניסויים עם batch size, קצבי למידה, ואופטימיזציה


📈 תוצאות

* מודל CNN פשוט: דיוק של **93.96%**
* מודל ResNet18 (Fine-Tuned): דיוק של **99.56%**
* ניסויים כללו בדיקת קצבי למידה, אופטימיזרים שונים, ו־augmentation
* התמודדות עם טעויות סיווג נובעת בעיקר מרקע מבלבל בתמונות


🧪 הצעות לשיפור עתידי

* הוספת **Attention Mechanisms** להתמקדות באובייקטים רלוונטיים
* הרחבת הדאטהסט למגוון רחב יותר של נופים
* שימוש בטכניקות חכמות יותר של **Augmentation** להתמודדות עם רעש חזותי

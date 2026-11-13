# iOS local portal fixtures

Captured `GET /state` responses from `mobilerun-ios --local` v0.5.0 (iPhone 17 Pro Max, iOS 27.2, 440x956 pt). The Settings captures have the Apple Account, follow-up, Wi-Fi, Cellular and Personal Hotspot rows removed and the second keyboard shown as English.

- `settings_root.json`: Settings at the top of its root list; the floating bottom search bar (Toolbar > SearchField) covers the Action Button and Appearance rows.
- `settings_search_keyboard.json`: Settings with the empty search field active, search suggestions shown and the keyboard up (search return key).
- `clock_add_alarm_sheet.json`: Clock Add Alarm sheet over the alarm list (one alarm, off); Label row (StaticText "Label" + TextField "Alarm") and Snooze row with its switch on.
- `clock_alarm_list.json`: Clock Alarms list with one alarm row (02:37, "Alarm") and its switch on.

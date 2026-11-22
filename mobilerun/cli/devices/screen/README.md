# Device screen

`mobilerun device screen screenshot` saves a PNG and prints its path.
`ui` prints the accessibility tree; `ui --raw` preserves its original indentation.
Both use the shared target selector and release their device connection afterward.

```sh
mobilerun device screen screenshot emulator-5554
```

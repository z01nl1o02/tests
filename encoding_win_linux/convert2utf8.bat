@echo off

setlocal enabledelayedexpansion

for /f %%n in (codes_with_chinese.txt) do (
::echo -ne '\xEF\xBB\xBF' > %%n.utf8bom
iconv -f GB2312 -t UTF-8 %%n > %%n.utf8
)

endlocal

@echo on

pause
pause
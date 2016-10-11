@echo off

setlocal enabledelayedexpansion

for /f %%n in (codes_with_chinese.txt) do (
::echo -ne '\xEF\xBB\xBF' > %%n.utf8bom
iconv -f UTF-8 -t GB2312 %%n > %%n.utf8bom
)

endlocal

@echo on


pause
pause
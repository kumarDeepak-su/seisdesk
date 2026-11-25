; Inno Setup script: wraps dist\SeisDesk (the PyInstaller folder) into one installer .exe.
; Compile with:  iscc /DAppVersion=0.12.0 packaging\seisdesk.iss

#ifndef AppVersion
  #define AppVersion "0.12.0"
#endif
#ifndef BuildSuffix
  #define BuildSuffix ""
#endif

[Setup]
AppId={{8C7F2A41-5B0E-4D3C-9A61-3E2F6B7D9C10}
AppName=SeisDesk
AppVersion={#AppVersion}
AppVerName=SeisDesk {#AppVersion}
AppPublisher=Deepak Kumar
DefaultDirName={autopf}\SeisDesk
DefaultGroupName=SeisDesk
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist
OutputBaseFilename=SeisDesk-{#AppVersion}{#BuildSuffix}-setup
SetupIconFile=seisdesk.ico
UninstallDisplayIcon={app}\SeisDesk.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\SeisDesk\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\dist\SeisDesk_User_Manual.pdf"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\SeisDesk"; Filename: "{app}\SeisDesk.exe"
Name: "{group}\SeisDesk User Manual"; Filename: "{app}\SeisDesk_User_Manual.pdf"; Check: ManualExists
Name: "{group}\Uninstall SeisDesk"; Filename: "{uninstallexe}"
Name: "{autodesktop}\SeisDesk"; Filename: "{app}\SeisDesk.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SeisDesk.exe"; Description: "{cm:LaunchProgram,SeisDesk}"; Flags: nowait postinstall skipifsilent

[Code]
function ManualExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\SeisDesk_User_Manual.pdf'));
end;

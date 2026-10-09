# NOVA unter Windows installieren

Gilt für Windows 10 und Windows 11, jeweils 64-Bit. Python oder Node.js sind **nicht** nötig.
Administratorrechte sind **nicht** nötig: NOVA wird nur für den angemeldeten Benutzer installiert.

> Prüfstatus: Der Installer wird in der GitHub-Actions-Pipeline `windows-release` auf
> `windows-latest` (Windows Server) gebaut und dort automatisch installiert, aktualisiert,
> repariert und deinstalliert (Bericht `install-test.md` in den Artefakten). Auf einem
> Windows-10/11-Desktop mit echter Benutzeroberfläche wurde er bisher **nicht** manuell geprüft –
> siehe `docs/test-report.md`.

## 1. Herunterladen und prüfen

1. Von der Release-Seite `NOVA-<Version>-x64.msi` und `SHA256SUMS.txt` herunterladen.
2. Prüfsumme vergleichen (PowerShell):

   ```powershell
   (Get-FileHash .\NOVA-0.1.0-x64.msi -Algorithm SHA256).Hash.ToLower()
   Get-Content .\SHA256SUMS.txt
   ```

   Die Werte müssen übereinstimmen. Das MSI ist **nicht code-signiert**. Windows SmartScreen
   warnt deshalb („Unbekannter Herausgeber“) – nur fortfahren, wenn die Prüfsumme stimmt.

## 2. Installieren

Doppelklick auf das MSI. Der Assistent zeigt:

1. Hinweis (Lizenzstand, Modelle, Ablage der Daten) – bestätigen.
2. Funktionen:
   * **NOVA** (immer): Programm und Startmenü-Einträge „NOVA“ und „Stop NOVA service“
   * **Desktop shortcut** (optional, standardmäßig aus)
   * **Start with Windows** (optional, standardmäßig aus): startet den Hintergrunddienst bei der Anmeldung
3. Installieren.

Unbeaufsichtigt (z. B. für IT-Verteilung):

```powershell
msiexec /i NOVA-0.1.0-x64.msi /qn /l*v install.log
msiexec /i NOVA-0.1.0-x64.msi /qn ADDLOCAL=Main,DesktopShortcut,Autostart /l*v install.log
```

Fehlende Voraussetzungen (32-Bit-Windows oder älter als Windows 10) melden sich mit einer klaren
Meldung, und die Installation bricht ab. Bei Problemen hilft das Protokoll von
`/l*v install.log`.

| Was | Wo |
|---|---|
| Programmdateien | `%LOCALAPPDATA%\Programs\NOVA` |
| Benutzerdaten (Einstellungen, Verläufe, Schlüssel, Logs, Modelle) | `%LOCALAPPDATA%\NOVA` |
| Startmenü | `NOVA`, `Stop NOVA service` |
| Version | „Apps & Features“ / „Programme und Features“, `nova --version`, Oberfläche (Status-Panel) |

## 3. Starten

Startmenü (oder optional Desktop) → **NOVA**. Es öffnet sich das **NOVA-Fenster**
(Desktop-App, `nova-desktop.exe`); einen Browser oder eine URL brauchst du nicht. Im
Hintergrund startet die App den NOVA-Core-Dienst. Er lauscht nur auf diesem Rechner
(`127.0.0.1:8765`).

Voraussetzung für das Fenster ist die **Microsoft Edge WebView2 Runtime**. Unter Windows 11
ist sie vorinstalliert, unter Windows 10 kommt sie in der Regel mit Edge. Fehlt sie, meldet
NOVA das und bietet an, die Oberfläche im Standardbrowser zu öffnen.

Beim Schließen des Fensters stoppt NOVA den Core-Dienst, wenn das Fenster ihn gestartet hat.
Wenn andere Programme (z. B. IC WARE HQ) NOVA nutzen sollen, aktiviere unter
**Status → Desktop app** „Keep the core service running when this window closes“. Dort gibt es
auch die Knöpfe **Restart core**, **Stop core and quit**, **Open in browser**, **Data folder**
und **Logs**.

Ohne Modell startet NOVA im **Setup-Modus**. Die Oberfläche zeigt „No local model available.“ und
den Knopf **Set up a model** (§4).

Stoppen: Startmenü → **Stop NOVA service**.

Kommandozeile: `%LOCALAPPDATA%\Programs\NOVA\nova.exe` (Ordner bei Bedarf in `PATH` aufnehmen):

```powershell
nova service status        # läuft? Port, Version, Log-Datei
nova service start | stop
nova open                  # Dienst starten + Browser
nova data path             # Datenordner anzeigen
```

Fehler beim Start erscheinen als Meldungsfenster. Details stehen in
`%LOCALAPPDATA%\NOVA\logs\desktop.log`, `launcher.log`, `service.log` und `nova.log`.

## 4. Modell einrichten

Modelle sind **nicht** im Installer enthalten. NOVA braucht zwei Dinge:

1. **Eine lokale Runtime**, die ein Modell bereitstellt: llama.cpp `llama-server` oder Ollama.
   Sie wird getrennt installiert, NOVA bringt keine mit.
2. **`%LOCALAPPDATA%\NOVA\models.toml`**: welches Modell über welche Runtime erreichbar ist.
   Eine Vorlage liegt unter `%LOCALAPPDATA%\Programs\NOVA\_internal\config\models.example.toml`.

Optional lädt NOVA GGUF-Dateien selbst herunter. Dazu kommen Einträge mit echter Adresse,
SHA-256 und Lizenz in `%LOCALAPPDATA%\NOVA\model-catalog.json`. Die Vorlage ist
`…\_internal\config\model-catalog.example.json`, deren Platzhalter bewusst abgelehnt werden.
NOVA erfindet keine Download-Adressen. Vor jedem Download zeigt NOVA:

* die Größe und den freien Speicherplatz (+5 % Reserve);
* den Arbeitsspeicher im Vergleich zu `min_ram_gb`: Ist zu wenig vorhanden, wird der Download blockiert;
* die Lizenz, die ausdrücklich bestätigt werden muss.

Downloads werden fortgesetzt, wenn sie unterbrochen wurden. Danach prüft NOVA die SHA-256 und
die GGUF-Signatur. Bei einer falschen Prüfsumme wird die Datei verworfen.

```powershell
nova models catalog
nova models plan <id>
nova models download <id> --accept-license
```

Danach `models.toml` anpassen und den Dienst neu starten:
`nova service stop` und dann `nova open`.

## 5. Aktualisieren

Das neuere MSI einfach installieren. Ein laufender Dienst wird vorher automatisch gestoppt. Die
alte Version wird entfernt und die neue installiert. `%LOCALAPPDATA%\NOVA` bleibt unverändert.
Ältere Versionen lassen sich nicht über neuere installieren: Es erscheint eine Meldung.

## 6. Reparieren

„Apps & Features“ → NOVA → Ändern → **Repair**, oder `msiexec /fa NOVA-<Version>-x64.msi`.
Gelöschte oder beschädigte Programmdateien werden ersetzt, Benutzerdaten bleiben unberührt.

## 7. Deinstallieren

„Apps & Features“ → NOVA → Deinstallieren, oder `msiexec /x NOVA-<Version>-x64.msi`.

* Der Dienst wird gestoppt.
* Programmdateien, Startmenü- und Desktop-Verknüpfungen sowie der Autostart werden entfernt.
* **Benutzerdaten bleiben erhalten** (`%LOCALAPPDATA%\NOVA`), damit eine Neuinstallation
  nahtlos weitermacht.

Daten bewusst löschen: **vor** der Deinstallation, bei gestopptem Dienst:

```powershell
nova service stop
nova data purge            # fragt nach und verlangt die Eingabe DELETE
```

Nach der Deinstallation kann der Ordner `%LOCALAPPDATA%\NOVA` auch von Hand gelöscht werden.

## 8. Andere Programme anbinden

Siehe `docs/integration-api.md`. Kurz: `nova integrations create --name <programm> --scope chat:complete`,
OpenAI-kompatibel unter `http://127.0.0.1:8765/v1`.

## 9. Bekannte Einschränkungen

* Kein Code-Signing (SmartScreen-Warnung) und keine automatische Aktualisierung.
* Der „Hintergrunddienst“ ist ein Benutzerprozess, kein Windows-Systemdienst. Er läuft nur,
  solange der Benutzer angemeldet ist.
* Die Repository-Lizenz fehlt noch. Der Installer zeigt deshalb einen Hinweis statt eines
  Lizenztextes.
* Runtimes (llama.cpp/Ollama) und Modelle werden nicht mitinstalliert.
* Ein Port-Konflikt auf 8765 wird gemeldet (der Dienst startet dann nicht). Ein anderer Port
  geht nur per Kommandozeile mit `--port`. Das Startmenü nutzt immer 8765.

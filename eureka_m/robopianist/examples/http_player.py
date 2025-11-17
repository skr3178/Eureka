# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An HTTP server that plays notes received through POST requests.

To try it out, start the server:
  python examples/http_player.py

Then send it a post request like:
  curl -X POST localhost:8080 -d 'ACTIVATION=[40,44]'
"""

import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np

from robopianist.music import midi_file, synthesizer
from robopianist.music.constants import NUM_KEYS

hostname = "localhost"
serverport = 8080

_ACTIVATION_RE = re.compile(r"^ACTIVATION=\[((\d+,)*(\d+)?)\]$")


class PianoServer(BaseHTTPRequestHandler):
    """An HTTP server that plays notes received through POST requests."""

    def __init__(self, *args, **kwargs):
        self._prev_activation = np.zeros(NUM_KEYS, dtype=bool)

        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Handle GET requests by showing usage information."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>HTTP Piano Player</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 20px auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        h1 { color: white; text-align: center; }
        .container {
            background: white;
            color: #333;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        .piano-container {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .piano-keys {
            display: flex;
            gap: 5px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .key {
            padding: 15px 20px;
            border: 2px solid #333;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.1s;
            user-select: none;
        }
        .key:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .key:active {
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .key.white {
            background: white;
            color: black;
        }
        .key.black {
            background: #333;
            color: white;
        }
        .key.pressed {
            background: #4CAF50 !important;
            color: white !important;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .stop-btn {
            background: #f44336;
            color: white;
        }
        .stop-btn:hover {
            background: #da190b;
        }
        .chord-btn {
            background: #2196F3;
            color: white;
        }
        .chord-btn:hover {
            background: #0b7dda;
        }
        .input-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        input[type="text"] {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            width: 300px;
        }
        .play-btn {
            background: #4CAF50;
            color: white;
        }
        .play-btn:hover {
            background: #45a049;
        }
        .status {
            text-align: center;
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>🎹 HTTP Piano Player</h1>
    <div class="container">
        <div class="piano-container">
            <h2>Interactive Piano</h2>
            <div id="status" class="status" style="display:none;"></div>
            
            <div class="piano-keys" id="piano">
                <!-- One octave of keys -->
                <div class="key white" data-key="40" data-note="C">C (40)</div>
                <div class="key black" data-key="41" data-note="C#">C# (41)</div>
                <div class="key white" data-key="42" data-note="D">D (42)</div>
                <div class="key black" data-key="43" data-note="D#">D# (43)</div>
                <div class="key white" data-key="44" data-note="E">E (44)</div>
                <div class="key white" data-key="45" data-note="F">F (45)</div>
                <div class="key black" data-key="46" data-note="F#">F# (46)</div>
                <div class="key white" data-key="47" data-note="G">G (47)</div>
                <div class="key black" data-key="48" data-note="G#">G# (48)</div>
                <div class="key white" data-key="49" data-note="A">A (49)</div>
                <div class="key black" data-key="50" data-note="A#">A# (50)</div>
                <div class="key white" data-key="51" data-note="B">B (51)</div>
            </div>
            
            <div class="controls">
                <button class="stop-btn" onclick="stopAll()">Stop All Notes</button>
                <button class="chord-btn" onclick="playChord([40,44,47])">C Major</button>
                <button class="chord-btn" onclick="playChord([40,43,47])">C Minor</button>
                <button class="chord-btn" onclick="playChord([40,44,47,51])">C Major 7</button>
            </div>
            
            <div class="input-group">
                <input type="text" id="customKeys" placeholder="Enter keys (e.g., 40,44,47)" />
                <button class="play-btn" onclick="playCustom()">Play Custom</button>
            </div>
        </div>
        
        <h2>API Usage</h2>
        <p>You can also send POST requests from the command line:</p>
        <pre>curl -X POST localhost:8080 -d 'ACTIVATION=[40,44,47]'</pre>
        <p><strong>Key range:</strong> 0-87 (88 piano keys) | <strong>Middle C:</strong> Key 40</p>
    </div>
    
    <script>
        let activeKeys = new Set();
        
        function showStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
            status.style.display = 'block';
            setTimeout(() => {
                status.style.display = 'none';
            }, 2000);
        }
        
        function sendActivation(keys) {
            const activation = 'ACTIVATION=[' + Array.from(keys).join(',') + ']';
            fetch('/', {
                method: 'POST',
                body: activation
            }).then(response => {
                if (response.ok) {
                    showStatus('Playing keys: ' + Array.from(keys).join(', '));
                } else {
                    showStatus('Error sending request', true);
                }
            }).catch(error => {
                showStatus('Connection error', true);
            });
            
            // Update UI
            document.querySelectorAll('.key').forEach(key => {
                const keyId = parseInt(key.dataset.key);
                if (keys.has(keyId)) {
                    key.classList.add('pressed');
                } else {
                    key.classList.remove('pressed');
                }
            });
        }
        
        function stopAll() {
            activeKeys.clear();
            sendActivation(activeKeys);
            showStatus('All notes stopped');
        }
        
        function playChord(keys) {
            activeKeys = new Set(keys);
            sendActivation(activeKeys);
        }
        
        function playCustom() {
            const input = document.getElementById('customKeys').value;
            const keys = input.split(',').map(k => parseInt(k.trim())).filter(k => !isNaN(k) && k >= 0 && k <= 87);
            if (keys.length > 0) {
                activeKeys = new Set(keys);
                sendActivation(activeKeys);
            } else {
                showStatus('Invalid key numbers', true);
            }
        }
        
        // Click to toggle keys
        document.querySelectorAll('.key').forEach(key => {
            key.addEventListener('click', () => {
                const keyId = parseInt(key.dataset.key);
                if (activeKeys.has(keyId)) {
                    activeKeys.delete(keyId);
                } else {
                    activeKeys.add(keyId);
                }
                sendActivation(activeKeys);
            });
        });
        
        // Keyboard support
        document.addEventListener('keydown', (e) => {
            const keyMap = {
                'a': 40, 'w': 41, 's': 42, 'e': 43, 'd': 44,
                'f': 45, 't': 46, 'g': 47, 'y': 48, 'h': 49,
                'u': 50, 'j': 51
            };
            if (keyMap[e.key] && !e.repeat) {
                activeKeys.add(keyMap[e.key]);
                sendActivation(activeKeys);
            }
            if (e.key === ' ') {
                e.preventDefault();
                stopAll();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            const keyMap = {
                'a': 40, 'w': 41, 's': 42, 'e': 43, 'd': 44,
                'f': 45, 't': 46, 'g': 47, 'y': 48, 'h': 49,
                'u': 50, 'j': 51
            };
            if (keyMap[e.key]) {
                activeKeys.delete(keyMap[e.key]);
                sendActivation(activeKeys);
            }
        });
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self) -> None:
        global _synth
        assert _synth is not None

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

        for line in self.rfile:
            m = _ACTIVATION_RE.match(str(line, "utf-8"))
            if m:
                activation = np.zeros(NUM_KEYS, dtype=bool)
                if m.group(1):
                    active = [int(d) for d in m.group(1).split(",")]
                    for key_id in active:
                        if key_id < NUM_KEYS:
                            activation[key_id] = True
                        else:
                            print(f"Invalid key id: {key_id}")

                state_change = activation ^ self._prev_activation

                # Note on events.
                for key_id in np.flatnonzero(state_change & ~self._prev_activation):
                    _synth.note_on(
                        midi_file.key_number_to_midi_number(key_id),
                        127,
                    )

                # Note off events.
                for key_id in np.flatnonzero(state_change & ~activation):
                    _synth.note_off(midi_file.key_number_to_midi_number(key_id))

                # Update state.
                self._prev_activation = activation.copy()

            break

    def log_request(self, request):
        del request  # Unused.


_synth: Optional[synthesizer.Synthesizer] = None

if __name__ == "__main__":
    _synth = synthesizer.Synthesizer()
    _synth.start()

    webServer = HTTPServer((hostname, serverport), PianoServer)
    print(f"Server started http://{hostname}:{serverport}")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    _synth.stop()
    print("Server stopped.")

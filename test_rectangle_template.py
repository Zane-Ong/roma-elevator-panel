import json
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import rectangle_server as server


class TemplatePersistenceTests(unittest.TestCase):
    def test_save_load_and_original_centers_preserved(self):
        original_panel=server.PANEL
        seed=json.loads((original_panel/'template.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=original_panel) as temporary:
            server.PANEL=Path(temporary)
            try:
                (server.PANEL/'template.json').write_text(json.dumps(seed),encoding='utf-8')
                client=TestClient(server.app)
                buttons=[]
                for b in seed['buttons']:
                    x,y=b['reference_xy']
                    buttons.append({**b,'rect_xyxy':[x-20,y-15,x+20,y+15]})
                result=client.post('/api/template',json={'sha256':seed['sha256'],'buttons':buttons})
                self.assertEqual(result.status_code,200,result.text)
                reloaded=TestClient(server.app).get('/api/template').json()
                self.assertTrue(reloaded['calibrated'])
                self.assertEqual(reloaded['buttons'],buttons)
                self.assertEqual(reloaded['sha256'],seed['sha256'])
                self.assertEqual(len(list((server.PANEL/'template_history').glob('*.json'))),1)
                self.assertEqual(client.post('/api/template',json={'sha256':'wrong','buttons':buttons}).status_code,400)
            finally:
                server.PANEL=original_panel
        self.assertEqual(json.loads((original_panel/'template.json').read_text(encoding='utf-8')),seed)


if __name__=='__main__':
    unittest.main()

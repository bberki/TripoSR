"""
TripoSR Batch Processor
Birden fazla görüntüyü otomatik olarak 3D modele çevirir

Kullanım:
    from batch_processor import BatchProcessor
    processor = BatchProcessor(model, output_dir="outputs")
    results = processor.process_batch(image_paths)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from PIL import Image
import gc
import time

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


class BatchProcessor:
    """
    TripoSR için batch işlem sınıfı
    
    Bu sınıf, birden fazla görüntüyü sırayla işleyerek
    3D mesh'lere dönüştürür ve sonuçları organize bir
    şekilde kaydeder.
    """
    
    def __init__(
        self,
        model: TSR,
        output_dir: str = "batch_outputs",
        device: str = "cuda:0"
    ):
        """
        BatchProcessor'ı başlat
        
        Args:
            model: Yüklenmiş TSR modeli
            output_dir: Çıktıların kaydedileceği ana klasör
            device: İşlem cihazı ('cuda:0', 'cuda:1', 'cpu')
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # rembg session'ı lazy loading için None başlat
        self._rembg_session = None
        
    @property
    def rembg_session(self):
        """Lazy loading için rembg session"""
        if self._rembg_session is None:
            import rembg
            self._rembg_session = rembg.new_session()
        return self._rembg_session
        
    def process_batch(
        self,
        image_paths: List[str],
        output_format: str = "obj",
        do_remove_background: bool = True,
        foreground_ratio: float = 0.85,
        mc_resolution: int = 256,
        save_processed_images: bool = True,
        continue_on_error: bool = True
    ) -> Dict:
        """
        Birden fazla görüntüyü sırayla işle
        
        Bu fonksiyon tüm işlem pipeline'ını yönetir:
        1. Batch klasörü oluşturur
        2. Her görüntüyü sırayla işler
        3. Hataları handle eder
        4. Sonuçları raporlar
        
        Args:
            image_paths: İşlenecek görüntü dosya yolları
            output_format: Mesh formatı ('obj', 'glb', 'ply', 'stl')
            do_remove_background: Arka planı kaldır (True/False)
            foreground_ratio: Nesnenin görüntüdeki oranı (0.5-1.0)
            mc_resolution: Marching Cubes grid çözünürlüğü (32-320)
            save_processed_images: İşlenmiş görüntüleri kaydet
            continue_on_error: Hata olunca devam et (True/False)
            
        Returns:
            Dict: {
                'batch_id': str,
                'batch_dir': str,
                'total': int,
                'successful': int,
                'failed': int,
                'results': List[Dict]
            }
        """
        # Batch klasörü oluştur (timestamp ile unique)
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.output_dir / batch_id
        batch_dir.mkdir(exist_ok=True, parents=True)
        
        # İstatistikler
        results = []
        successful = 0
        failed = 0
        total_time = 0
        
        # Başlangıç mesajları
        print("\n" + "="*70)
        print(f"🚀 BATCH İŞLEM BAŞLATILDI")
        print("="*70)
        print(f"📅 Batch ID    : {batch_id}")
        print(f"📁 Output Dir  : {batch_dir}")
        print(f"📊 Toplam      : {len(image_paths)} görüntü")
        print(f"🎨 Format      : {output_format.upper()}")
        print(f"🖼️  Resolution  : {mc_resolution}x{mc_resolution}x{mc_resolution}")
        print(f"🎭 Remove BG   : {'Evet' if do_remove_background else 'Hayır'}")
        print("="*70 + "\n")
        
        # Her görüntüyü işle (progress bar ile)
        for idx, img_path in enumerate(tqdm(image_paths, desc="🔄 İşleniyor", 
                                            unit="img", ncols=100)):
            try:
                # Görüntü bilgileri
                img_path_obj = Path(img_path)
                img_name = img_path_obj.stem
                img_extension = img_path_obj.suffix
                
                # Her görüntü için alt klasör
                output_subdir = batch_dir / f"{idx+1:03d}_{img_name}"
                output_subdir.mkdir(exist_ok=True, parents=True)
                
                # Tek görüntüyü işle
                result = self._process_single_image(
                    img_path=img_path,
                    img_name=img_name,
                    output_subdir=output_subdir,
                    output_format=output_format,
                    do_remove_background=do_remove_background,
                    foreground_ratio=foreground_ratio,
                    mc_resolution=mc_resolution,
                    save_processed_images=save_processed_images
                )
                
                # Sonucu kaydet
                if result["status"] == "success":
                    successful += 1
                    total_time += result.get("processing_time", 0)
                    
                    results.append({
                        "index": idx + 1,
                        "filename": img_name + img_extension,
                        "input_path": str(img_path),
                        "output_dir": str(output_subdir),
                        "mesh_path": result["mesh_path"],
                        "status": "✅ success",
                        "processing_time_sec": round(result["processing_time"], 2),
                        "vertices": result.get("vertices", 0),
                        "faces": result.get("faces", 0)
                    })
                    
                    # Başarılı işlem log
                    tqdm.write(f"  ✅ [{idx+1}/{len(image_paths)}] {img_name} "
                              f"({result['processing_time']:.1f}s)")
                else:
                    failed += 1
                    results.append({
                        "index": idx + 1,
                        "filename": img_name + img_extension,
                        "input_path": str(img_path),
                        "status": "❌ failed",
                        "error": result.get("error", "Unknown error")
                    })
                    
                    # Hata log
                    tqdm.write(f"  ❌ [{idx+1}/{len(image_paths)}] {img_name} "
                              f"- {result.get('error', 'Unknown')}")
                    
                    # Hata olunca dur
                    if not continue_on_error:
                        print("\n⚠️  Hata nedeniyle işlem durduruluyor...")
                        break
                        
            except Exception as e:
                failed += 1
                error_msg = str(e)
                
                results.append({
                    "index": idx + 1,
                    "filename": Path(img_path).name,
                    "input_path": str(img_path),
                    "status": "❌ error",
                    "error": error_msg
                })
                
                tqdm.write(f"  💥 [{idx+1}/{len(image_paths)}] Kritik Hata: {error_msg}")
                
                if not continue_on_error:
                    print("\n⚠️  Kritik hata nedeniyle işlem durduruluyor...")
                    break
            
            # Her işlem sonrası memory temizle
            self._cleanup_memory()
        
        # Ortalama işlem süresi
        avg_time = total_time / successful if successful > 0 else 0
        
        # Rapor oluştur ve kaydet
        summary = {
            "batch_id": batch_id,
            "batch_dir": str(batch_dir),
            "total": len(image_paths),
            "processed": successful + failed,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/(successful+failed)*100):.1f}%" if (successful+failed) > 0 else "0%",
            "total_time_sec": round(total_time, 2),
            "avg_time_per_image_sec": round(avg_time, 2),
            "results": results
        }
        
        self._save_batch_report(batch_dir, summary)
        
        # Özet rapor
        print("\n" + "="*70)
        print(f"🎉 BATCH İŞLEM TAMAMLANDI")
        print("="*70)
        print(f"Toplam      : {len(image_paths)} görüntü")
        print(f"Başarılı    : {successful} ({summary['success_rate']})")
        print(f"Başarısız   : {failed}")
        print(f"Toplam Süre : {total_time:.1f} saniye")
        print(f"Ortalama    : {avg_time:.1f} saniye/görüntü")
        print(f"Çıktılar    : {batch_dir}")
        print("="*70 + "\n")
        
        return summary
    
    def _process_single_image(
        self,
        img_path: str,
        img_name: str,
        output_subdir: Path,
        output_format: str,
        do_remove_background: bool,
        foreground_ratio: float,
        mc_resolution: int,
        save_processed_images: bool
    ) -> Dict:
        """
        Tek bir görüntüyü işle
        
        İşlem Adımları:
        1. Görüntü yükleme
        2. Ön işleme (background removal, resize)
        3. Model inference (triplane generation)
        4. Mesh extraction (marching cubes)
        5. Dosya kaydetme
        
        Args:
            img_path: Görüntü dosya yolu
            img_name: Görüntü adı (uzantısız)
            output_subdir: Çıktıların kaydedileceği klasör
            output_format: Mesh formatı
            do_remove_background: Arka plan kaldırma
            foreground_ratio: Ön plan oranı
            mc_resolution: Marching cubes çözünürlüğü
            save_processed_images: İşlenmiş görüntüyü kaydet
            
        Returns:
            Dict: İşlem sonucu (status, paths, timing, stats)
        """
        start_time = time.time()
        
        try:
            # 1. GÖRÜNTÜ YÜKLEME
            image = Image.open(img_path)
            original_size = image.size
            
            # 2. ÖN İŞLEME
            if do_remove_background:
                # RGB'ye çevir
                image = image.convert("RGB")
                
                # Arka planı kaldır (U2-Net model)
                image = remove_background(image, self.rembg_session)
                
                # Foreground'u yeniden boyutlandır
                image = resize_foreground(image, foreground_ratio)
                
                # Alpha channel'ı beyaz arka plana blend et
                image_np = np.array(image).astype(np.float32) / 255.0
                image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + \
                           (1 - image_np[:, :, 3:4]) * 0.5
                image = Image.fromarray((image_np * 255.0).astype(np.uint8))
            else:
                # RGBA ise alpha'yı blend et
                if image.mode == "RGBA":
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + \
                               (1 - image_np[:, :, 3:4]) * 0.5
                    image = Image.fromarray((image_np * 255.0).astype(np.uint8))
            
            # İşlenmiş görüntüyü kaydet
            if save_processed_images:
                processed_img_path = output_subdir / "processed_input.png"
                image.save(processed_img_path)
            
            # 3. MODEL INFERENCE
            with torch.no_grad():
                # Triplane generation
                scene_codes = self.model([image], device=self.device)
                
                # Mesh extraction
                meshes = self.model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=mc_resolution
                )
            
            mesh = meshes[0]
            
            # 4. MESH KAYDETME
            mesh_path = output_subdir / f"mesh.{output_format}"
            mesh.export(str(mesh_path))
            
            # Mesh istatistikleri
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            
            # İşlem süresi
            processing_time = time.time() - start_time
            
            # Metadata kaydet
            metadata = {
                "input_image": str(img_path),
                "original_size": original_size,
                "output_format": output_format,
                "mc_resolution": mc_resolution,
                "vertices": num_vertices,
                "faces": num_faces,
                "processing_time_sec": round(processing_time, 2),
                "background_removed": do_remove_background,
                "foreground_ratio": foreground_ratio
            }
            
            metadata_path = output_subdir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "mesh_path": str(mesh_path),
                "processing_time": processing_time,
                "vertices": num_vertices,
                "faces": num_faces
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _cleanup_memory(self):
        """
        GPU ve sistem belleğini temizle
        
        Her görüntü işlendikten sonra çağrılır.
        Memory leak'leri önler.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _save_batch_report(self, batch_dir: Path, summary: Dict):
        """
        Batch işlem raporunu JSON olarak kaydet
        
        Args:
            batch_dir: Batch klasörü
            summary: Özet bilgiler ve sonuçlar
        """
        report_path = batch_dir / "batch_report.json"
        
        # Timestamp ekle
        summary["timestamp"] = datetime.now().isoformat()
        summary["report_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Kısa özet de oluştur
        summary_path = batch_dir / "SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("TRIPOSR BATCH PROCESSING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Batch ID         : {summary['batch_id']}\n")
            f.write(f"Total Images     : {summary['total']}\n")
            f.write(f"Successful       : {summary['successful']}\n")
            f.write(f"Failed           : {summary['failed']}\n")
            f.write(f"Success Rate     : {summary['success_rate']}\n")
            f.write(f"Total Time       : {summary['total_time_sec']}s\n")
            f.write(f"Average Time     : {summary['avg_time_per_image_sec']}s per image\n")
            f.write(f"\n{'='*70}\n\n")
            
            # Başarılı işlemler
            if summary['successful'] > 0:
                f.write("SUCCESSFUL PROCESSES:\n")
                f.write("-" * 70 + "\n")
                for result in summary['results']:
                    if result['status'] == '✅ success':
                        f.write(f"  [{result['index']:3d}] {result['filename']}\n")
                        f.write(f"        Time: {result.get('processing_time_sec', 0)}s\n")
                        f.write(f"        Vertices: {result.get('vertices', 0):,}\n")
                        f.write(f"        Faces: {result.get('faces', 0):,}\n\n")
            
            # Başarısız işlemler
            if summary['failed'] > 0:
                f.write("\n" + "="*70 + "\n")
                f.write("FAILED PROCESSES:\n")
                f.write("-" * 70 + "\n")
                for result in summary['results']:
                    if result['status'] != '✅ success':
                        f.write(f"  [{result['index']:3d}] {result['filename']}\n")
                        f.write(f"        Error: {result.get('error', 'Unknown')}\n\n")


def get_images_from_folder(folder_path: str, recursive: bool = False) -> List[str]:
    """
    Klasördeki tüm görüntü dosyalarını bul
    
    Args:
        folder_path: Aranacak klasör
        recursive: Alt klasörleri de ara
        
    Returns:
        List[str]: Görüntü dosya yolları (sıralı)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Bu bir klasör değil: {folder_path}")
    
    # Desteklenen uzantılar
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', 
                       '.bmp', '.BMP', '.webp', '.WEBP'}
    
    image_paths = []
    
    if recursive:
        # Alt klasörleri de tara
        for ext in image_extensions:
            image_paths.extend(folder.rglob(f"*{ext}"))
    else:
        # Sadece ana klasör
        for ext in image_extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
    
    # Sırala ve string'e çevir
    return sorted([str(p) for p in image_paths])

#!/usr/bin/env python3
"""
TripoSR Batch İşlem CLI
Birden fazla görüntüyü toplu olarak 3D modele çevirir

Kullanım Örnekleri:
    # Basit
    python run_batch.py ./my_images
    
    # Gelişmiş
    python run_batch.py ./my_images --output results --format glb --mc-resolution 320
    
    # Alt klasörler dahil
    python run_batch.py ./my_images --recursive
    
    # CPU modunda
    python run_batch.py ./my_images --device cpu
"""

import argparse
import sys
import torch
from pathlib import Path

from tsr.system import TSR
from batch_processor import BatchProcessor, get_images_from_folder


def parse_arguments():
    """Komut satırı argümanlarını parse et"""
    parser = argparse.ArgumentParser(
        description="🎨 TripoSR Batch Processor - Toplu 3D Model Üretimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Basit kullanım
  %(prog)s ./input_images
  
  # Özel ayarlarla
  %(prog)s ./input_images --output my_outputs --format glb --mc-resolution 320
  
  # Alt klasörleri de tara
  %(prog)s ./input_images --recursive
  
  # Arka plan kaldırmadan
  %(prog)s ./input_images --no-remove-bg
  
  # CPU modunda (CUDA yoksa)
  %(prog)s ./input_images --device cpu

Desteklenen Formatlar:
  Görüntü: .png, .jpg, .jpeg, .bmp, .webp
  Mesh   : obj, glb, ply, stl
        """
    )
    
    # ZORUNLU ARGÜMANLAR
    parser.add_argument(
        "input_folder",
        type=str,
        help="Görüntülerin bulunduğu klasör yolu"
    )
    
    # ÇIKTI AYARLARI
    output_group = parser.add_argument_group('Çıktı Ayarları')
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="batch_outputs",
        help="Çıktıların kaydedileceği klasör (varsayılan: batch_outputs)"
    )
    output_group.add_argument(
        "--format", "-f",
        type=str,
        default="obj",
        choices=["obj", "glb", "ply", "stl"],
        help="Mesh çıktı formatı (varsayılan: obj)"
    )
    output_group.add_argument(
        "--save-processed-images",
        action="store_true",
        help="İşlenmiş görüntüleri de kaydet"
    )
    
    # GÖRÜNTÜ İŞLEME AYARLARI
    image_group = parser.add_argument_group('Görüntü İşleme Ayarları')
    image_group.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Arka plan kaldırmayı devre dışı bırak"
    )
    image_group.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Ön plan oranı, 0.5-1.0 arası (varsayılan: 0.85)"
    )
    image_group.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Alt klasörleri de tara"
    )
    
    # MODEL AYARLARI
    model_group = parser.add_argument_group('Model Ayarları')
    model_group.add_argument(
        "--mc-resolution",
        type=int,
        default=256,
        help="Marching Cubes çözünürlüğü, 32-320 arası (varsayılan: 256)"
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="İşlem cihazı: cuda:0, cuda:1, cpu (varsayılan: cuda:0)"
    )
    model_group.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk boyutu, memory yönetimi için (varsayılan: 8192)"
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        default="stabilityai/TripoSR",
        help="Model yolu (varsayılan: stabilityai/TripoSR)"
    )
    
    # DİĞER AYARLAR
    other_group = parser.add_argument_group('Diğer Ayarlar')
    other_group.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Hata olunca işlemi durdur (varsayılan: devam et)"
    )
    other_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal çıktı (sadece önemli mesajlar)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Argümanları doğrula"""
    errors = []
    
    # Input klasör kontrolü
    if not Path(args.input_folder).exists():
        errors.append(f"❌ Input klasörü bulunamadı: {args.input_folder}")
    
    # Foreground ratio kontrolü
    if not (0.5 <= args.foreground_ratio <= 1.0):
        errors.append(f"❌ Foreground ratio 0.5-1.0 arasında olmalı: {args.foreground_ratio}")
    
    # MC resolution kontrolü
    if not (32 <= args.mc_resolution <= 320):
        errors.append(f"❌ MC resolution 32-320 arasında olmalı: {args.mc_resolution}")
    
    # Chunk size kontrolü
    if args.chunk_size < 0:
        errors.append(f"❌ Chunk size pozitif olmalı: {args.chunk_size}")
    
    if errors:
        print("\n".join(errors))
        sys.exit(1)


def print_banner():
    """Başlangıç banner'ı"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                    🎨 TripoSR Batch Processor 🎨                    ║
║                                                                      ║
║            Tek Fotoğraftan 3D Model - Toplu İşlem Aracı            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Ana fonksiyon"""
    
    # Argümanları parse et
    args = parse_arguments()
    
    # Argümanları doğrula
    validate_arguments(args)
    
    # Banner göster
    if not args.quiet:
        print_banner()
    
    # Device kontrolü ve ayarlama
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("⚠️  CUDA kullanılamıyor, CPU'ya geçiliyor...")
            args.device = "cpu"
        else:
            # CUDA device sayısını kontrol et
            device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
            if device_id >= torch.cuda.device_count():
                print(f"⚠️  CUDA:{device_id} bulunamadı, CUDA:0'a geçiliyor...")
                args.device = "cuda:0"
    
    # Konfigürasyon özeti
    if not args.quiet:
        print("\n" + "="*70)
        print("📋 KONFİGÜRASYON")
        print("="*70)
        print(f"📂 Input Klasörü       : {args.input_folder}")
        print(f"📁 Output Klasörü      : {args.output}")
        print(f"📊 Mesh Formatı        : {args.format.upper()}")
        print(f"🖼️  MC Resolution       : {args.mc_resolution}")
        print(f"💻 Device              : {args.device.upper()}")
        print(f"📦 Chunk Size          : {args.chunk_size}")
        print(f"🎭 Remove Background   : {'Hayır' if args.no_remove_bg else 'Evet'}")
        print(f"📏 Foreground Ratio    : {args.foreground_ratio}")
        print(f"🔄 Recursive           : {'Evet' if args.recursive else 'Hayır'}")
        print(f"🛑 Stop on Error       : {'Evet' if args.stop_on_error else 'Hayır'}")
        print("="*70 + "\n")
    
    # Görüntüleri bul
    try:
        print("🔍 Görüntüler aranıyor...")
        image_paths = get_images_from_folder(args.input_folder, recursive=args.recursive)
        
        if not image_paths:
            print(f"\n❌ '{args.input_folder}' klasöründe görüntü bulunamadı!")
            print("   Desteklenen formatlar: .png, .jpg, .jpeg, .bmp, .webp")
            sys.exit(1)
        
        print(f"✅ {len(image_paths)} görüntü bulundu")
        
        if not args.quiet:
            print("\nİlk 5 görüntü:")
            for i, path in enumerate(image_paths[:5], 1):
                print(f"  {i}. {Path(path).name}")
            if len(image_paths) > 5:
                print(f"  ... ve {len(image_paths)-5} tane daha\n")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        sys.exit(1)
    
    # Onay iste (10'dan fazla görüntü varsa)
    if len(image_paths) > 10 and not args.quiet:
        response = input(f"\n⚠️  {len(image_paths)} görüntü işlenecek. Devam edilsin mi? (E/h): ")
        if response.lower() not in ['e', 'evet', 'y', 'yes', '']:
            print("❌ İşlem iptal edildi.")
            sys.exit(0)
    
    # Modeli yükle
    try:
        print("\n🔄 Model yükleniyor...")
        model = TSR.from_pretrained(
            args.model_path,
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        
        # Chunk size ayarla
        model.renderer.set_chunk_size(args.chunk_size)
        
        # Device'a taşı
        model.to(args.device)
        
        print(f"✅ Model yüklendi! ({args.device})")
        
        # Model bilgileri
        if not args.quiet:
            if args.device.startswith("cuda"):
                gpu_name = torch.cuda.get_device_name(args.device)
                gpu_memory = torch.cuda.get_device_properties(args.device).total_memory / 1e9
                print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
    except Exception as e:
        print(f"\n❌ Model yükleme hatası: {e}")
        sys.exit(1)
    
    # Batch processor oluştur
    processor = BatchProcessor(
        model=model,
        output_dir=args.output,
        device=args.device
    )
    
    # Batch işlemi başlat
    try:
        results = processor.process_batch(
            image_paths=image_paths,
            output_format=args.format,
            do_remove_background=not args.no_remove_bg,
            foreground_ratio=args.foreground_ratio,
            mc_resolution=args.mc_resolution,
            save_processed_images=args.save_processed_images,
            continue_on_error=not args.stop_on_error
        )
        
        # Başarı durumu
        if results['successful'] == results['total']:
            print("\n🎉 Tüm görüntüler başarıyla işlendi!")
            sys.exit(0)
        elif results['successful'] > 0:
            print(f"\n⚠️  Kısmi başarı: {results['successful']}/{results['total']} görüntü işlendi")
            sys.exit(0)
        else:
            print("\n❌ Hiçbir görüntü işlenemedi!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

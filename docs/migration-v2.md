# Guia de Migração: v1.x → v2.0

## O que mudou (BREAKING CHANGE)

O modelo v2.0.0 exige **128 features** de entrada em vez de 64.

## Impacto no firmware

```cpp
// v1.x (ANTIGO — NÃO funciona com v2.0)
float input[64];   // janela de 64 amostras

// v2.0 (NOVO — obrigatório)
float input[128];  // janela de 128 amostras
```

## Passos de migração

1. Atualizar o buffer de leitura do acelerômetro para 128 amostras
2. Substituir o arquivo .tflite por vibration_classifier_v2.tflite
3. Recompilar e 'reflashear' o firmware no ESP32
4. Validar as 3 classes de saída (0=normal, 1=iminente, 2=falha)

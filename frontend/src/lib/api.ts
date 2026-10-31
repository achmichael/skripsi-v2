import { API_BASE, IS_MOCK } from "./config";
import type { PredictionInput, PredictionOutput, HealthResponse } from "@/schemas/types";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || `Request failed: ${res.status}`);
  }

  return res.json();
}

// Mock responses for UI development without backend
function mockPredict(input: PredictionInput): PredictionOutput {
  const nominalFactor = input.nominal_token / 100000;
  const baseDays = input.capacity === "900" ? 12 : input.capacity === "1300" ? 10 : 8;
  const predicted = Math.round((baseDays * nominalFactor + Math.random() * 3) * 100) / 100;
  return {
    predicted_days: Math.max(1, predicted),
    capacity: input.capacity,
    model_version: "mock-v1.0",
  };
}

function mockHealth(): HealthResponse {
  return {
    status: "ok",
    model_version: "mock-v1.0",
    capacities_available: ["900", "1300", "2200"],
  };
}

export const api = {
  health: async (): Promise<HealthResponse> => {
    if (IS_MOCK) return mockHealth();
    return request("/health");
  },

  predict: async (input: PredictionInput): Promise<PredictionOutput> => {
    if (IS_MOCK) {
      await new Promise((r) => setTimeout(r, 800 + Math.random() * 700));
      return mockPredict(input);
    }
    return request("/predict", {
      method: "POST",
      body: JSON.stringify({
        capacity: input.capacity,
        kota: input.kota,
        jumlah_anggota: input.jumlah_anggota,
        status_subsidi: input.status_subsidi,
        nominal_token: input.nominal_token,
        frekuensi_isi: input.frekuensi_isi,
        kulkas_jumlah: input.kulkas_jumlah,
        kulkas_kategori: input.kulkas_kategori,
        tv_jumlah: input.tv_jumlah,
        tv_kategori: input.tv_kategori,
        ac_jumlah: input.ac_jumlah,
        ac_pk: input.ac_pk,
        ac_kategori: input.ac_kategori,
        kipas_jumlah: input.kipas_jumlah,
        kipas_kategori: input.kipas_kategori,
        rice_cooker_jumlah: input.rice_cooker_jumlah,
        rice_cooker_kategori: input.rice_cooker_kategori,
        mesin_cuci_jumlah: input.mesin_cuci_jumlah,
        mesin_cuci_kategori: input.mesin_cuci_kategori,
      }),
    });
  },
};

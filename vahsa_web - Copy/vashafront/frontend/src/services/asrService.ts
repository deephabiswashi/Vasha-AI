// ASR Service for connecting frontend to backend
const API_BASE_URL = 'http://localhost:8000';

interface ScriptConfig {
  script: string;
  fontFamily: string;
  forceNativeScript: boolean;
}

export const SCRIPT_CONFIG: Record<string, ScriptConfig> = {
  bn: {
    script: 'Bengali',
    fontFamily: 'Noto Sans Bengali',
    forceNativeScript: true
  },
  hi: {
    script: 'Devanagari',
    fontFamily: 'Noto Sans Devanagari',
    forceNativeScript: true
  },
  ta: {
    script: 'Tamil',
    fontFamily: 'Noto Sans Tamil',
    forceNativeScript: true
  },
  te: {
    script: 'Telugu',
    fontFamily: 'Noto Sans Telugu',
    forceNativeScript: true
  }
};

export interface ASRResponse {
  success: boolean;
  transcription: string;
  language: string;
  language_name: string;
  model_used: string;
  message: string;
  error?: string;
}

export interface Language {
  [key: string]: string;
}

export interface Model {
  id: string;
  name: string;
  description: string;
  supports_fallback: boolean;
  fallback_to?: string;
}

export interface ModelsResponse {
  models: Model[];
  message: string;
}

export interface LanguagesResponse {
  languages: Language;
  message: string;
}

// Add language to script mapping at the top
const LANGUAGE_SCRIPT_MAP: Record<string, string> = {
  hi: 'Devanagari',
  bn: 'Bengali',
  ta: 'Tamil',
  te: 'Telugu'
};

class ASRService {
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`ASR API Error (${endpoint}):`, error);
      throw error;
    }
  }

  async getLanguages(): Promise<LanguagesResponse> {
    return this.makeRequest<LanguagesResponse>('/languages');
  }

  async getModels(): Promise<ModelsResponse> {
    return this.makeRequest<ModelsResponse>('/asr/models');
  }

  async processFileUpload(
    file: File,
    model: string = 'whisper',
    whisperSize: string = 'large',
    decoding: string = 'ctc'
  ): Promise<ASRResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('whisper_size', whisperSize);
    formData.append('decoding', decoding);

    return this.makeRequest<ASRResponse>('/asr/upload', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }

  async processYouTubeAudio(
    youtubeUrl: string,
    model: string = 'whisper',
    whisperSize: string = 'large',
    decoding: string = 'ctc'
  ): Promise<ASRResponse> {
    const formData = new FormData();
    formData.append('youtube_url', youtubeUrl);
    formData.append('model', model);
    formData.append('whisper_size', whisperSize);
    formData.append('decoding', decoding);

    return this.makeRequest<ASRResponse>('/asr/youtube', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for FormData
      body: formData,
    });
  }

  async processMicrophoneAudio(
    audioBlob: Blob,
    model: string = 'whisper',
    whisperSize: string = 'large',
    decoding: string = 'ctc',
    duration: number = 5
  ): Promise<ASRResponse> {
    // Convert blob to file for upload
    const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
    
    return this.processFileUpload(audioFile, model, whisperSize, decoding);
  }

  // Helper method to check if backend is available
  async checkBackendHealth(): Promise<boolean> {
    try {
      await this.getLanguages();
      return true;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  async transcribeYoutube(
    youtubeUrl: string, 
    opts?: { 
      model?: string; 
      whisper_size?: string; 
      decoding?: string; 
    }
  ): Promise<ASRResponse> {
    const formData = new FormData();
    formData.append('youtube_url', youtubeUrl);

    // Keep user's model choice but enforce script settings
    formData.append('model', opts?.model || 'whisper');
    formData.append('whisper_size', opts?.whisper_size || 'large');
    formData.append('decoding', opts?.decoding || 'ctc');

    try {
      const response = await fetch(`${API_BASE_URL}/asr/youtube`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Post-process response to ensure correct script
      if (data.language && SCRIPT_CONFIG[data.language]) {
        const config = SCRIPT_CONFIG[data.language];
        data.script = config.script;
        // If using whisper, ensure text is in native script
        if (data.model_used === 'whisper' && config.forceNativeScript) {
          console.log(`Ensuring native script for ${data.language}`);
        }
      }

      console.log('ASR Response:', {
        language: data.language,
        model_used: data.model_used,
        script: data.script
      });

      return data;
    } catch (error) {
      console.error('ASR processing error:', error);
      throw error;
    }
  }
}

export const asrService = new ASRService();

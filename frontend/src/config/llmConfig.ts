// LLM configuration stored in localStorage
export interface LLMConfig {
  base_url: string
  api_key: string
  model: string
}

const STORAGE_KEY = 'llm_config'

export const DEFAULT_CONFIG: LLMConfig = {
  base_url: 'https://api.deepseek.com/v1',
  api_key: '',
  model: 'deepseek-chat',
}

export const PRESET_PROVIDERS = [
  { name: 'DeepSeek', base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  { name: 'Kimi (Moonshot)', base_url: 'https://api.moonshot.cn/v1', model: 'moonshot-v1-8k' },
  { name: 'OpenAI', base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  { name: 'Claude (via proxy)', base_url: 'https://api.anthropic.com/v1', model: 'claude-3-5-haiku-20241022' },
]

export function loadConfig(): LLMConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) }
  } catch {}
  return { ...DEFAULT_CONFIG }
}

export function saveConfig(config: LLMConfig): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
}

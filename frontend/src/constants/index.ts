import AnthropicIcon from '@/assets/models/anthropic.webp';
import AzureOpenAiIcon from '@/assets/models/azure-openai.png';
import BaiChuanIcon from '@/assets/models/baichuan.jpeg';
import ChatGPTIcon from '@/assets/models/chat-gpt.png';
import ChatglmIcon from '@/assets/models/chatglm.png';
import DeepseekIcon from '@/assets/models/deepseek.svg';
import FalconIcon from '@/assets/models/falcon.png';
import GeminiIcon from '@/assets/models/gemini.svg';
import Glm4Icon from '@/assets/models/glm4.png';
import GorillaIcon from '@/assets/models/gorilla.png';
import GuanacoIcon from '@/assets/models/guanaco.png';
import JinaIcon from '@/assets/models/jina.png';
import OpenrouterIcon from '@/assets/models/openrouter.png';
import QianwenIcon from '@/assets/models/qianwen.png';
import VicunaIcon from '@/assets/models/vicuna.jpg';
import WenxinYiyanIcon from '@/assets/models/wenxinyiyan.png';
import XAiIcon from '@/assets/models/xai.svg';

import AlibabaIcon from '@/assets/models/alibabacloud.svg';
import SiliconflowIcon from '@/assets/models/siliconflow.svg';

import EmailIcon from '@/assets/collection_source/email.png';
import FeishuIcon from '@/assets/collection_source/feishu.png';
import FtpIcon from '@/assets/collection_source/ftp_icon.png';
import GithubIcon from '@/assets/collection_source/github.png';
import LocalPathIcon from '@/assets/collection_source/local.png';
import OssIcon from '@/assets/collection_source/oss.png';
import S3Icon from '@/assets/collection_source/s3.png';
import SystemIcon from '@/assets/collection_source/system.png';
import UrlIcon from '@/assets/collection_source/url.png';

import GmailIcon from '@/assets/collection_source/gmail.png';
import OutlookIcon from '@/assets/collection_source/outlook.png';
import QQGmailIcon from '@/assets/collection_source/qq.png';

import iconAgent from '@/assets/bots/agent.svg';
import iconCommon from '@/assets/bots/common.svg';
import iconKnowledge from '@/assets/bots/knowledge.svg';

import {
  BotTypeEnum,
  CollectionConfig,
  CollectionStatusEnum,
  DocumentFulltextIndexStatusEnum,
  DocumentGraphIndexStatusEnum,
  DocumentStatusEnum,
  DocumentVectorIndexStatusEnum,
} from '@/api';
import type { CollectionConfigSource, CollectionEmailSource } from '@/types';

export * from './theme';

export const CSS_PREFIX = 'ape';
export const TOPBAR_HEIGHT = 50;
export const SIDEBAR_WIDTH = 65;
export const NAVIGATION_WIDTH = 220;

export const LOCALES = [
  {
    key: 'en-US',
    label: 'English',
  },
  {
    key: 'zh-CN',
    label: '简体中文',
  },
];

export const DATETIME_FORMAT = 'YYYY-MM-DD HH:mm:ss';

export const MODEL_PROVIDER_ICON: {
  [key in string]: string;
} = {
  xai: XAiIcon,
  gemini: GeminiIcon,
  anthropic: AnthropicIcon,
  openai: ChatGPTIcon,
  deepseek: DeepseekIcon,
  alibabacloud: AlibabaIcon,
  siliconflow: SiliconflowIcon,
  openrouter: OpenrouterIcon,
  jina: JinaIcon,

  vicuna: VicunaIcon,
  chatglm: ChatglmIcon,
  chatgpt: ChatGPTIcon,
  baichuan: BaiChuanIcon,
  wenxinyiyan: WenxinYiyanIcon,
  'azure-openai': AzureOpenAiIcon,
  guanaco: GuanacoIcon,
  falcon: FalconIcon,
  gorilla: GorillaIcon,
  qianwen: QianwenIcon,
  'glm-4': Glm4Icon,
};

export const BOT_TYPE_ICON: { [key in BotTypeEnum]: React.ReactNode } = {
  knowledge: iconKnowledge,
  common: iconCommon,
  agent: iconAgent,
};

export const COLLECTION_SOURCE: {
  [key in CollectionConfigSource]: {
    icon: React.ReactNode;
    enabled?: boolean;
  };
} = {
  system: {
    icon: SystemIcon,
    enabled: true,
  },
  local: {
    icon: LocalPathIcon,
  },
  s3: {
    icon: S3Icon,
  },
  oss: {
    icon: OssIcon,
  },
  ftp: {
    icon: FtpIcon,
  },
  email: {
    icon: EmailIcon,
  },
  url: {
    icon: UrlIcon,
  },
  git: {
    icon: GithubIcon,
  },
  feishu: {
    icon: FeishuIcon,
  },
};

export const COLLECTION_SOURCE_EMAIL: {
  [key in CollectionEmailSource]: {
    icon: string;
    pop_server: string;
    port: number;
  };
} = {
  gmail: {
    icon: GmailIcon,
    pop_server: 'pop.gmail.com',
    port: 995,
  },
  outlook: {
    icon: OutlookIcon,
    pop_server: 'outlook.office365.com',
    port: 995,
  },
  qqmail: {
    icon: QQGmailIcon,
    pop_server: 'pop.qq.com',
    port: 995,
  },
  others: {
    icon: EmailIcon,
    pop_server: '',
    port: 995,
  },
};

export const DOCUMENT_DEFAULT_CONFIG: CollectionConfig = {
  source: 'system',
  enable_knowledge_graph: true,
  enable_summary: false,
  crontab: {
    enabled: false,
    minute: '0',
    hour: '0',
    day_of_month: '*',
    month: '*',
    day_of_week: '*',
  },
};

export const UI_COLLECTION_STATUS: {
  [key in CollectionStatusEnum]:
    | 'success'
    | 'processing'
    | 'error'
    | 'default'
    | 'warning';
} = {
  ACTIVE: 'success',
  INACTIVE: 'error',
  DELETED: 'error',
  SUMMARY_GENERATING: 'processing',
};

export const UI_DOCUMENT_STATUS: {
  [key in DocumentStatusEnum]:
    | 'success'
    | 'processing'
    | 'error'
    | 'default'
    | 'warning';
} = {
  PENDING: 'warning',
  RUNNING: 'processing',
  FAILED: 'error',
  COMPLETE: 'success',
  DELETED: 'default',
  DELETING: 'warning',
};

export const UI_INDEX_STATUS: {
  [key in
    | DocumentVectorIndexStatusEnum
    | DocumentFulltextIndexStatusEnum
    | DocumentGraphIndexStatusEnum
    | string]: 'success' | 'processing' | 'error' | 'default' | 'warning'; // 新增，兼容 summary
} = {
  PENDING: 'warning',
  CREATING: 'processing',
  ACTIVE: 'success',
  DELETING: 'warning',
  DELETION_IN_PROGRESS: 'processing',
  FAILED: 'error',
  SKIPPED: 'default',
};

export const SUPPORTED_DOC_EXTENSIONS = [
  '.pdf',
  '.doc',
  '.docx',
  '.ppt',
  '.pptx',
  '.xls',
  '.xlsx',
  '.epub',
  '.md',
  '.ipynb',
  '.txt',
  '.htm',
  '.html',
];
export const SUPPORTED_MEDIA_EXTENSIONS = [
  '.jpg',
  '.jpeg',
  '.png',
  '.webm',
  '.mp3',
  '.mp4',
  '.mpeg',
  '.mpga',
  '.m4a',
  '.wav',
];
export const SUPPORTED_COMPRESSED_EXTENSIONS = [
  '.zip',
  '.rar',
  '.r00',
  '.7z',
  '.tar',
  '.gz',
  '.xz',
  '.bz2',
  '.tar.gz',
  '.tar.xz',
  '.tar.bz2',
  '.tar.7z',
];

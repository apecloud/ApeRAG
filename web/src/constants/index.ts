import BaiChuanIcon from '@/assets/baichuan.jpeg';
import ChatGPTIcon from '@/assets/chat-gpt.png';
import ChatglmIcon from '@/assets/chatglm.png';
import AzureOpenAiIcon from '@/assets/azure-openai.png';
import VicunaIcon from '@/assets/vicuna.jpg';
import WenxinYiyanIcon from '@/assets/wenxinyiyan.png';
import GuanacoIcon from '@/assets/guanaco.png';
import FalconIcon from '@/assets/falcon.png';
import GorillaIcon from '@/assets/gorilla.png';
import QianwenIcon from '@/assets/qianwen.png';

import IconLocalPath from '@/assets/localpath.svg';
import IconLocalPathOn from '@/assets/localpath-on.svg';
import IconUpload from '@/assets/icon-upload.svg';
import IconUploadOn from '@/assets/upload-on.svg';
import IconFTP from '@/assets/ftp.svg';
import IconFTPOn from '@/assets/ftp-on.svg';
import IconEmail from '@/assets/email.svg';
import IconEmailOn from '@/assets/email-on.svg';
import IconAWSS3 from '@/assets/s3.svg';
import IconAWSS3On from '@/assets/s3-on.svg';
import IconOSS from '@/assets/oss.svg';
import IconOSSOn from '@/assets/oss-on.svg';
import IconFeishu from '@/assets/feishu.svg';
import IconFeishuOn from '@/assets/feishu-on.svg';
import IconLink from '@/assets/link.svg';
import IconLinkOn from '@/assets/link-on.svg';
import IconGithub from '@/assets/github.svg';
import IconGithubOn from '@/assets/github-on.svg';
import IconDeepSeek from '@/assets/deepseek.svg';

import {
  TypesCollectionConfig,
  TypesCollectionConfigSource,
  TypesCollectionConfigSourceOption,
  TypesCollectionStatus,
  TypesDocumentStatus,
  TypesEmailSource,
  TypesSocketStatus,
} from '@/types';
import { PresetStatusColorType } from 'antd/es/_util/colors';
import { ReadyState } from 'react-use-websocket';

export const DOCUMENT_DEFAULT_CONFIG: TypesCollectionConfig = {
  source: 'system',
  crontab: {
    enabled: false,
    minute: '0',
    hour: '0',
    day_of_month: '*',
    month: '*',
    day_of_week: '*',
  },
};

export const BOT_DEFAULT_CONFIG = {
  llm: {
    similarity_score_threshold: 0.5,
    similarity_topk: 3,
    context_window: 2000,
    prompt_template: '',
    temperature: 0.3,
  },
};

export const COLLECTION_MODEL_OPTIONS_CONFIG: {
  [key in string]: {
    icon?: string;
  };
} = {
  'vicuna': {
    icon: VicunaIcon,
  },
  'deepseek': {
    icon: IconDeepSeek,
  },
  'chatglm': {
    icon: ChatglmIcon,
  },
  'glm-4': {
    icon: ChatglmIcon,
  },
  'chatgpt': {
    icon: ChatGPTIcon,
  },
  'baichuan': {
    icon: BaiChuanIcon,
  },
  'wenxinyiyan': {
    icon: WenxinYiyanIcon,
  },
  'azure-openai':{
    icon: AzureOpenAiIcon,
  },
  'guanaco':{
    icon: GuanacoIcon,
  },
  'falcon':{
    icon: FalconIcon,
  },
  'gorilla':{
    icon: GorillaIcon,
  },
  'qianwen':{
    icon: QianwenIcon,
  }
};

export const CODE_DEFAULT_CONFIG = {};

export const DOCUMENT_SOURCE_OPTIONS: TypesCollectionConfigSourceOption[] = [
  {
    label: 'Local File',
    value: 'system',
    icon: IconUpload,
    icon_on: IconUploadOn,
  },
  {
    label: 'AWS S3',
    value: 's3',
    icon: IconAWSS3,
    icon_on: IconAWSS3On,
  },
  {
    label: 'Aliyun OSS',
    value: 'oss',
    icon: IconOSS,
    icon_on: IconOSSOn,
  },
  {
    label: 'FTP',
    value: 'ftp',
    icon: IconFTP,
    icon_on: IconFTPOn,
  },
  {
    label: 'Email',
    value: 'email',
    icon: IconEmail,
    icon_on: IconEmailOn,
  },
  {
    label: 'URL',
    value: 'url',
    icon: IconLink,
    icon_on: IconLinkOn,
  },
  {
    label: 'Github',
    value: 'github',
    icon: IconGithub,
    icon_on: IconGithubOn,
  },
  {
    label: 'Feishu',
    value: 'feishu',
    icon: IconFeishu,
    icon_on: IconFeishuOn,
  },
];

if (NODE_ENV === 'development') {
  DOCUMENT_SOURCE_OPTIONS.splice(1, 0, {
    label: 'Local Path',
    value: 'local',
    icon: IconLocalPath,
    icon_on: IconLocalPathOn,
  });
}

export const getSourceItemByValue = (s?: TypesCollectionConfigSource) => {
  const data = DOCUMENT_SOURCE_OPTIONS.find((item) => item.value === s);
  return data;
};

export const COLLECTION_STATUS_TAG_COLORS: {
  [key in TypesCollectionStatus]: PresetStatusColorType;
} = {
  INACTIVE: 'error',
  ACTIVE: 'success',
  DELETED: 'error',
};

export const DOCUMENT_STATUS_TAG_COLORS: {
  [key in TypesDocumentStatus]: PresetStatusColorType;
} = {
  PENDING: 'warning',
  RUNNING: 'processing',
  FAILED: 'error',
  COMPLETE: 'success',
  DELETED: 'default',
};

export const SOCKET_STATUS_MAP: { [key in ReadyState]: TypesSocketStatus } = {
  [ReadyState.CONNECTING]: 'Connecting',
  [ReadyState.OPEN]: 'Open',
  [ReadyState.CLOSING]: 'Closing',
  [ReadyState.CLOSED]: 'Closed',
  [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
};

export const CODE_PROMOT_EXAMPLE = [
  {
    title: 'Snake Game',
    text: 'Build a console-based Snake game implemented in C++.',
  },
  {
    title: 'Currency Converter',
    text: 'Build a currency converter app using an API for exchange rates.Use HTML, CSS, and JavaScript for the frontend and Node.js for the backend. Allow users to convert between different currencies.',
  },
  {
    title: 'File Organizer',
    text: 'Create a file organizer CLI tool in Golang that sorts files in a directory based on their file types (e.g., images, documents, audio) and moves them into corresponding folders.',
  },
];

export const EMAIL_CONNECT_INFO: {
  [key in TypesEmailSource]: {
    pop_server: string;
    port: number;
    title: string;
    tips: string[];
  };
} = {
  gmail: {
    pop_server: 'pop.gmail.com',
    port: 995,
    title: 'Please follow the steps below to connect to your Gmail account',
    tips: [
      'Enable POP service in the Gmail’s web application',
      'In Google account, enable 2-step verification',
      'Create your google account app password for Gmail, which is not account password.',
      'Enter your Gmaill address and app password',
    ],
  },
  outlook: {
    pop_server: 'outlook.office365.com',
    port: 995,
    title:
      'Please follow the steps below to connect to your Outlook email account',
    tips: [
      'Enable POP service in the Outlook email’s web application',
      'Enter your email address and account password',
    ],
  },
  qqmail: {
    pop_server: 'pop.qq.com',
    port: 995,
    title: 'Please follow the steps below to connect to your QQMail account',
    tips: [
      'Enable POP service in the QQMail’s web application',
      'Get the authorization code, which is not account password.',
      'Enter your email address and authorization code',
    ],
  },
  others: {
    pop_server: '',
    port: 995,
    title: 'Please follow the steps below to connect to your email account',
    tips: [
      'Enable POP service in the email’s web application',
      'If your email has POP authorization code, generate it.',
      'Enter your email’s POP server and port',
      'Enter your email address and password or authorization code',
    ],
  },
};

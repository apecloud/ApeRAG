import { bot } from './zh-CN/bots';
import { collection } from './zh-CN/collection';
import { model, model_provider } from './zh-CN/models';
import { user, users } from './zh-CN/users';

export default {
  ...user,
  ...users,
  ...bot,
  ...model,
  ...model_provider,
  ...collection,

  'text.welcome': '欢迎使用ApeRAG',
  'text.authorizing': '授权中',
  'text.authorize.error': '认证失败',
  'text.authorize.error.description': '联系管理员，检查系统权限配置',
  'text.pageNotFound': '当前页面不存在',
  'text.emailConfirm':
    '您的邮箱账号当前未验证，请查看邮箱邮件点击确认进行身份验证，验证通过后刷新页面重试。',
  'text.system.builtin': '系统内置',
  'text.empty': '未查询到相关数据',
  'text.tags': '标签',
  'text.createdAt': '创建时间',
  'text.updatedAt': '更新时间',
  'text.status': '状态',
  'text.history.records': '历史记录',
  'text.direction': '方向',
  'text.direction.TB': '从上到下',
  'text.direction.LR': '从左到右',
  'text.references': '引用',
  'text.integrations': '外部集成',
  'text.title': '标题',
  'text.title.required': '标题为必填项',
  'text.description': '描述',
  'text.trial': '免费试用',

  tips: '提示',
  'tips.delete.success': '删除成功',
  'tips.create.success': '创建成功',
  'tips.update.success': '更新成功',
  'tips.upload.success': '上传成功',
  'tips.upload.error': '上传失败',

  cloud: '----------------------------------',
  'cloud.region': 'Region地址',
  'cloud.region.placeholder': 'Region地址',
  'cloud.region.required': 'Region地址为必填项',
  'cloud.authorize': '授权',
  'cloud.authorize.access_key_id': 'Access Key',
  'cloud.authorize.access_key_id.required': 'Access Key 为必填项',
  'cloud.authorize.secret_access_key': 'Secret Access Key',
  'cloud.authorize.secret_access_key.required': 'Secret Access Key 为必填项',
  'cloud.bucket': 'Bucket',
  'cloud.bucket.placeholder': 'Bucket名称',
  'cloud.bucket.required': 'Bucket名称为必填项',
  'cloud.bucket.add': '添加Bucket',
  'cloud.bucket.directory': '目录地址',
  'cloud.bucket.directory.required': '目录地址为必填项',

  ftp: '----------------------------------',
  'ftp.service_address': '服务地址',
  'ftp.service_address.host': 'FTP地址',
  'ftp.service_address.host.required': 'FTP地址为必填项目',
  'ftp.service_address.port': '端口',
  'ftp.service_address.port.required': '端口为必填项目',
  'ftp.authorize': '权限',
  'ftp.authorize.username': 'FTP用户名',
  'ftp.authorize.username.required': 'FTP用户名为必填项',
  'ftp.authorize.password': 'FTP密码',
  'ftp.authorize.password.required': 'FTP密码为必填项',
  'ftp.path': '文件路径',
  'ftp.path.required': '文件路径为必填项',

  email: '----------------------------------',
  'email.source': '邮箱类型',
  'email.pop_server': '服务地址',
  'email.pop_server.url': 'POP3 / SMTP地址',
  'email.pop_server.url.required': '地址为必填项',
  'email.pop_server.port': '端口',
  'email.pop_server.port.required': '端口为必填项',
  'email.authorize': '授权',
  'email.authorize.email_address': '邮箱',
  'email.authorize.email_address.invalid': '邮箱格式不正确',
  'email.authorize.email_address.required': '邮箱为必填项',
  'email.authorize.email_password': '密码',
  'email.authorize.email_password.required': '密码为必填项',

  'email.gmail': 'Gmail',
  'email.gmail.tips.title':
    'Please follow the steps below to connect to your Gmail account',
  'email.gmail.tips.description': `
1. Enable POP service in the Gmail’s web application
2. In Google account, enable 2-step verification
3. Create your google account app password for Gmail, which is not account password.
4. Enter your Gmaill address and app password
  `,
  'email.qqmail': 'QQ邮箱',
  'email.qqmail.tips.title':
    'Please follow the steps below to connect to your QQMail account',
  'email.qqmail.tips.description': `
1. Enable POP service in the QQMail’s web application
2. Get the authorization code, which is not account password.
3. Enter your email address and authorization code
  `,
  'email.outlook': 'Outlook',
  'email.outlook.tips.title':
    'Please follow the steps below to connect to your Outlook email account',
  'email.outlook.tips.description': `
1. Enable POP service in the Outlook email’s web application
2. Enter your email address and account password
`,
  'email.others': '其他邮箱',
  'email.others.tips.title':
    'Please follow the steps below to connect to your email account',
  'email.others.tips.description': `
1. Enable POP service in the email’s web application
2. If your email has POP authorization code, generate it.
3. Enter your email’s POP server and port
4. Enter your email address and password or authorization code
`,

  feishu: '----------------------------------',
  'feishu.authorize': '权限',
  'feishu.authorize.app_id': 'App ID',
  'feishu.authorize.app_id.required': 'App ID 为必填项',
  'feishu.authorize.app_secret': 'App Secret',
  'feishu.authorize.app_secret.required': 'App Secret 为必填项',
  'feishu.doc_space': '文档空间',
  'feishu.doc_space.space_id': 'Space ID',
  'feishu.doc_space.space_id.required': 'Space ID 为必填项',
  'feishu.doc_space.node_id': 'Node ID',
  'feishu.doc_space.node_id.required': 'Node ID 为必填项',

  git: '----------------------------------',
  'git.repo': '仓库地址',
  'git.repo.required': '仓库地址为必填项',
  'git.branch': '分支',
  'git.branch.required': '分支为必填项',
  'git.path': '路径',
  'git.path.required': '路径为必填项',

  action: '----------------------',
  'action.name': '操作',
  'action.search': '搜索',
  'action.backToHome': '返回首页',
  'action.signin': '登录',
  'action.signout': '登出',
  'action.back': '返回',
  'action.delete': '删除',
  'action.sync': '同步',
  'action.settings': '设置',
  'action.fitView': '1:1视图',
  'action.save': '保存',
  'action.update': '更新',
  'action.ok': '确定',
  'action.cancel': '取消',
  'action.close': '关闭',
  'action.run': '运行',
  'action.debug': '调试',
  'action.refresh': '刷新',
  'action.confirm': '确认',
  'action.rename': '重命名',

  document: '文档',
  'document.upload': '上传文档',
  'document.delete.confirm': '文档 "{name}" 将会被删除，确定此操作吗？',
  'document.name': '文件名',
  'document.size': '文件大小',
  'document.status': '状态',
  'document.status.PENDING': 'Pending',
  'document.status.RUNNING': '运行中',
  'document.status.FAILED': '失败',
  'document.status.COMPLETE': '已完成',
  'document.status.DELETED': '已删除',
  'document.status.DELETING': '删除中',

  flow: '任务流',
  'flow.settings': '任务流',
  'flow.edge.smoothstep': '折线',
  'flow.edge.bezier': 'Bezier曲线',
  'flow.node.add': '添加节点',

  chat: '---------------',
  'chat.all': '全部会话',
  'chat.new': '新会话',
  'chat.start_new': '开启新会话',
  'chat.delete': '删除会话',
  'chat.title': '会话名称',
  'chat.title_required': '请输入会话名称',
  'chat.input_placeholder': '给 {title} 发送消息',
  'chat.delete.confirm': '确定删除此会话吗？',
  'chat.empty_description':
    '我可以帮你写代码、读文件、写作各种创意内容，请把你的任务交给我吧~',

  system: '------------------------------',
  'system.management': '系统设置',
};

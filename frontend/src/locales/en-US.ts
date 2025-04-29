import { apiKeys } from './en-US/apiKeys';
import { bot } from './en-US/bots';
import { collection } from './en-US/collection';
import { model, model_provider } from './en-US/models';
import { user, users } from './en-US/users';

export default {
  ...user,
  ...users,
  ...bot,
  ...model,
  ...model_provider,
  ...collection,
  ...apiKeys,

  'text.welcome': 'Welcome to ApeRAG',
  'text.authorizing': 'Authorizing',
  'text.authorize.error': 'Authorization',
  'text.authorize.error.description':
    'Contact the administrator to check the system authorization configuration',
  'text.pageNotFound': 'This is not the web page you are looking for.',
  'text.emailConfirm':
    'Your email account is currently not verified. Please check your email and click Confirm to verify your identity. After the verification is passed, refresh the page and try again.',
  'text.system.builtin': 'Built-in',
  'text.empty': 'No data found',
  'text.tags': 'Tags',
  'text.createdAt': 'Creation time',
  'text.updatedAt': 'Update time',
  'text.lastUsedAt': 'Last used time',
  'text.status': 'Status',
  'text.history.records': 'History',
  'text.direction': 'Direction',
  'text.direction.TB': 'Top to bottom',
  'text.direction.LR': 'Left to right',
  'text.references': 'References',
  'text.integrations': 'Integrations',
  'text.title': 'Title',
  'text.title.required': 'Title is required',
  'text.description': 'Description',
  'text.trial': 'Free Trial',

  action: '----------------------',
  'action.name': 'Actions',
  'action.search': 'Search',
  'action.backToHome': 'Home',
  'action.signin': 'Sign in',
  'action.signout': 'Sign out',
  'action.back': 'Back',
  'action.delete': 'Delete',
  'action.sync': 'Sync',
  'action.settings': 'Settings',
  'action.fitView': '1:1',
  'action.save': 'Save',
  'action.update': 'Update',
  'action.ok': 'OK',
  'action.cancel': 'Cancel',
  'action.close': 'Close',
  'action.run': 'Run',
  'action.debug': 'Debug',
  'action.refresh': 'Refresh',
  'action.confirm': 'Confirm',
  'action.rename': 'Rename',

  tips: '--------------------------',
  'tips.delete.success': 'Delete successful',
  'tips.create.success': 'Create successful',
  'tips.update.success': 'Update successful',
  'tips.upload.success': 'Upload successful',
  'tips.upload.error': 'Upload error',

  cloud: '----------------------------------',
  'cloud.region': 'Region',
  'cloud.region.placeholder': 'Region',
  'cloud.region.required': 'Region is required',
  'cloud.authorize': 'Authorization',
  'cloud.authorize.access_key_id': 'Access Key',
  'cloud.authorize.access_key_id.required': 'Access Key is required',
  'cloud.authorize.secret_access_key': 'Secret Access Key',
  'cloud.authorize.secret_access_key.required': 'Secret Access Key is required',
  'cloud.bucket': 'Bucket',
  'cloud.bucket.placeholder': 'Bucket name',
  'cloud.bucket.required': 'Bucket name is required',
  'cloud.bucket.add': 'Add bucket',
  'cloud.bucket.directory': 'Directory',
  'cloud.bucket.directory.required': 'Directory is required',

  ftp: '----------------------------------',
  'ftp.service_address': 'Service address',
  'ftp.service_address.host': 'FTP address',
  'ftp.service_address.host.required': 'FTP address is required',
  'ftp.service_address.port': 'Port',
  'ftp.service_address.port.required': 'Port is required',
  'ftp.authorize': 'Authorization',
  'ftp.authorize.username': 'Username',
  'ftp.authorize.username.required': 'Username is required',
  'ftp.authorize.password': 'Password',
  'ftp.authorize.password.required': 'Password is required',
  'ftp.path': 'File path',
  'ftp.path.required': 'File path is required',

  email: '----------------------------------',
  'email.source': 'Email source',
  'email.pop_server': 'Service address',
  'email.pop_server.url': 'POP3 / SMTP',
  'email.pop_server.url.required': 'Service address is required',
  'email.pop_server.port': 'Port',
  'email.pop_server.port.required': 'Service port is required',
  'email.authorize': 'Authorization',
  'email.authorize.email_address': 'Email',
  'email.authorize.email_address.invalid': 'Email verification failed',
  'email.authorize.email_address.required': 'Email address is required',
  'email.authorize.email_password': 'Password',
  'email.authorize.email_password.required': 'Email password is required',

  'email.gmail': 'Gmail',
  'email.gmail.tips.title':
    'Please follow the steps below to connect to your Gmail account',
  'email.gmail.tips.description': `
1. Enable POP service in the Gmail's web application
2. In Google account, enable 2-step verification
3. Create your google account app password for Gmail, which is not account password.
4. Enter your Gmaill address and app password
`,
  'email.qqmail': 'QQMail',
  'email.qqmail.tips.title':
    'Please follow the steps below to connect to your QQMail account',
  'email.qqmail.tips.description': `
1. Enable POP service in the QQMail's web application
2. Get the authorization code, which is not account password.
3. Enter your email address and authorization code
`,
  'email.outlook': 'Outlook',
  'email.outlook.tips.title':
    'Please follow the steps below to connect to your Outlook email account',
  'email.outlook.tips.description': `
1. Enable POP service in the Outlook email's web application
2. Enter your email address and account password
`,
  'email.others': 'Others',
  'email.others.tips.title':
    'Please follow the steps below to connect to your email account',
  'email.others.tips.description': `
1. Enable POP service in the email's web application
2. If your email has POP authorization code, generate it.
3. Enter your email's POP server and port
4. Enter your email address and password or authorization code
`,

  feishu: '----------------------------------',
  'feishu.authorize': 'Authorization',
  'feishu.authorize.app_id': 'App ID',
  'feishu.authorize.app_id.required': 'App ID is required',
  'feishu.authorize.app_secret': 'App Secret',
  'feishu.authorize.app_secret.required': 'App Secret is required',
  'feishu.doc_space': 'Space / Node',
  'feishu.doc_space.space_id': 'Space ID',
  'feishu.doc_space.space_id.required': 'Space ID is required',
  'feishu.doc_space.node_id': 'Node ID',
  'feishu.doc_space.node_id.required': 'Node ID is required',

  git: '----------------------------------',
  'git.repo': 'Github Repo',
  'git.repo.required': 'Github Repo is required',
  'git.branch': 'Branch',
  'git.branch.required': 'Branch is required',
  'git.path': 'Path',
  'git.path.required': 'Path is required',

  document: '--------------------------',
  'document.upload': 'Upload',
  'document.delete.confirm':
    'The document "{name}" will be deleted, confirm the current operation.',
  'document.name': 'Name',
  'document.size': 'Size',
  'document.status': 'Status',
  'document.status.PENDING': 'Pending',
  'document.status.RUNNING': 'Running',
  'document.status.FAILED': 'Failed',
  'document.status.COMPLETE': 'Completed',
  'document.status.DELETED': 'Deleted',
  'document.status.DELETING': 'Deleting',

  flow: '---------------',
  'flow.name': 'Workflow',
  'flow.settings': 'Workflow',
  'flow.edge.smoothstep': 'Polyline',
  'flow.edge.bezier': 'Bezier',
  'flow.node.add': 'Add node',

  chat: '---------------',
  'chat.all': 'All chats',
  'chat.new': 'New chat',
  'chat.start_new': 'Start a new chat',
  'chat.delete': 'Delete chat',
  'chat.title': 'Title',
  'chat.title_required': 'Title is required',
  'chat.input_placeholder': 'Send a message to {title}',
  'chat.delete.confirm': 'Are you sure you want to delete this conversation?',
  'chat.empty_description':
    'I can help you write code, read files, and create various creative content. Just hand over your tasks to me!',

  system: '------------------------------',
  'system.management': 'Settings',

  feedback: '---------------',
  'feedback.bad.title': 'What\'s wrong with this response?',
  'feedback.bad.description': 'Please select a reason for your feedback',
  'feedback.message.placeholder': 'Please provide more details about your feedback',
  'feedback.tag.harmful': 'Harmful / Unsafe Content',
  'feedback.tag.unsafe': 'Unsafe Content',
  'feedback.tag.fake': 'False or Misleading Information',
  'feedback.tag.unhelpful': 'Not Helpful',
  'feedback.tag.other': 'Other Issue',
};

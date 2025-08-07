// import { PageContainer } from '@/components';
// import { ThemeType } from 'ahooks/lib/useTheme';
// import { useEffect, useState } from 'react';
// import { Link, useModel } from 'umi';
// import imgSnapshot from '../assets/page/snapshot.png';
// import styles from './home.less';

// export default function HomePage() {
//   const { themeName, setThemeName } = useModel('global');
//   const [realThemeName] = useState<ThemeType>(themeName);

//   useEffect(() => {
//     setThemeName('dark');
//     return () => setThemeName(realThemeName);
//   }, [themeName]);

//   return (
//     <PageContainer>
//       <div className={styles.banner}>
//         <p className={styles.summary}>
//           人人都会用 • 支持多种文档 • 可私有化部署
//         </p>
//         <h1 className={styles.title1}>用最简单的方式</h1>
//         <h2 className={styles.title2}>拥有知识库 24h 智能答疑机器人</h2>
//         <Link to="/bots" className={styles.btn}>
//           快速开始
//         </Link>
//       </div>
//       <div style={{ textAlign: 'center' }}>
//         <img src={imgSnapshot} style={{ maxWidth: '100%' }} />
//       </div>
//     </PageContainer>
//   );
// }

import { Navigate } from 'umi';

export default () => <Navigate to="/marketplace" replace />;

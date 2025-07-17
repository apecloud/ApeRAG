import { ThemeType } from 'ahooks/lib/useTheme';
import { AliasToken } from 'antd/es/theme/internal';

// basic tokens for all mode
const BASIC_TOKENS: Partial<AliasToken> = {
  borderRadius: 4,
  colorPrimary: '#0173E6', // '#7C49FF',
  fontWeightStrong: 400,
};

// dark mode token
const DARK_TOKENS: Partial<AliasToken> = {
  colorBgLayout: '#0F1214',
  colorBgContainer: '#131719',
  colorBgElevated: '#131719', // dropdown背景
  colorBgContainerDisabled: '#232526', // input disabled
  colorBgSpotlight: 'rgba(0, 0, 0, 0.85)', // tooltip bg for dark mode

  controlItemBgActive: '#1C2022',

  colorTextDisabled: '#55595E',
  colorTextPlaceholder: '#55595E',
  colorTextSecondary: '#878B92',
  colorTextDescription: '#878B92',

  colorBorder: '#242B30', // input border
  colorBorderSecondary: '#1C2126', // card border
};

// light mode token
const LIGHT_TOKENS: Partial<AliasToken> = {
  colorBgLayout: '#F7F7F9',
  colorBgContainer: '#FFF',
  colorBgElevated: '#FFF',
  colorBgSpotlight: 'rgba(0, 0, 0, 0.85)', // tooltip bg for light mode
  controlItemBgActive: '#E8E8EA',
  colorBorder: '#CAD0D8', // input border
  colorBorderSecondary: '#E7EAEE', // card border
};

export const THEME_TOKENS: {
  [key in ThemeType]: Partial<AliasToken>;
} = {
  dark: {
    ...BASIC_TOKENS,
    ...DARK_TOKENS,
  },
  light: {
    ...BASIC_TOKENS,
    ...LIGHT_TOKENS,
  },
};

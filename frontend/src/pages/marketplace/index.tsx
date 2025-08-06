import { PageContainer, PageHeader } from '@/components';
import { Col, Input, Pagination, Result, Row, theme } from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { toast } from 'react-toastify';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { history, useIntl, useModel } from 'umi';
import { CollectionMarketplaceCard } from './CollectionMarketplaceCard';

export default () => {
  const { formatMessage } = useIntl();
  const {
    marketplaceCollections,
    marketplaceLoading,
    marketplacePagination,
    getMarketplaceCollections,
    subscribeToCollection,
    unsubscribeFromCollection,
  } = useModel('collection');
  const { user } = useModel('user');
  const { token } = theme.useToken();

  const [searchKeyword, setSearchKeyword] = useState<string>('');
  const [subscribing, setSubscribing] = useState<string>(''); // Track which collection is being subscribed

  // Load marketplace collections on page mount
  useEffect(() => {
    getMarketplaceCollections(1, 12);
  }, []);

  const handleSearchChange = useCallback((value: string) => {
    setSearchKeyword(value);
  }, []);

  const handlePageChange = useCallback(
    (page: number, pageSize?: number) => {
      getMarketplaceCollections(page, pageSize || 12);
    },
    [getMarketplaceCollections],
  );

  const handleCardClick = useCallback(
    (collection: any) => {
      if (!user) return;

      if (collection.owner_user_id === user.id) {
        // Owner clicks - go to owner's collection page
        history.push(`/collections/${collection.id}/documents`);
      } else {
        // All other users can access marketplace collection page directly
        history.push(`/marketplace/collections/${collection.id}`);
      }
    },
    [user],
  );

  const handleSubscribe = useCallback(
    async (collectionId: string) => {
      if (!collectionId) return false;

      setSubscribing(collectionId);
      try {
        const success = await subscribeToCollection(collectionId);
        if (success) {
          toast.success(
            formatMessage({ id: 'collection.marketplace.subscribe.success' }),
          );
          return true;
        } else {
          toast.error(
            formatMessage({ id: 'collection.marketplace.subscribe.failed' }),
          );
          return false;
        }
      } catch (error) {
        toast.error(
          formatMessage({ id: 'collection.marketplace.subscribe.failed' }),
        );
        return false;
      } finally {
        setSubscribing('');
      }
    },
    [subscribeToCollection, formatMessage],
  );

  const handleUnsubscribe = useCallback(
    async (collectionId: string) => {
      if (!collectionId) return false;

      setSubscribing(collectionId);
      try {
        const success = await unsubscribeFromCollection(collectionId);
        if (success) {
          toast.success(
            formatMessage({ id: 'collection.marketplace.unsubscribe.success' }),
          );
          return true;
        } else {
          toast.error(
            formatMessage({ id: 'collection.marketplace.unsubscribe.failed' }),
          );
          return false;
        }
      } catch (error) {
        toast.error(
          formatMessage({ id: 'collection.marketplace.unsubscribe.failed' }),
        );
        return false;
      } finally {
        setSubscribing('');
      }
    },
    [unsubscribeFromCollection, formatMessage],
  );

  // Filter collections by search keyword
  const filteredCollections = useMemo(() => {
    return (
      marketplaceCollections?.filter(
        (collection) =>
          !searchKeyword ||
          collection.title
            ?.toLowerCase()
            .includes(searchKeyword.toLowerCase()) ||
          collection.description
            ?.toLowerCase()
            .includes(searchKeyword.toLowerCase()) ||
          collection.owner_username
            ?.toLowerCase()
            .includes(searchKeyword.toLowerCase()),
      ) || []
    );
  }, [marketplaceCollections, searchKeyword]);

  const renderEmptyState = () => {
    if (searchKeyword && filteredCollections.length === 0) {
      return (
        <Result
          icon={<UndrawEmpty primaryColor={token.colorPrimary} height="240px" />}
          title={formatMessage({ id: 'common.search.no.results' })}
          subTitle={formatMessage({
            id: 'common.search.try.different.keywords',
          })}
        />
      );
    }

    if (!marketplaceCollections || marketplaceCollections.length === 0) {
      return (
        <Result
          icon={<UndrawEmpty primaryColor={token.colorPrimary} height="240px" />}
          title={formatMessage({ id: 'collection.marketplace.empty.title' })}
          subTitle={formatMessage({
            id: 'collection.marketplace.empty.description',
          })}
        />
      );
    }
    return null;
  };

  return (
    <PageContainer loading={marketplaceLoading && !marketplaceCollections}>
      <PageHeader
        title={formatMessage({ id: 'collection.marketplace.title' })}
        description={formatMessage({
          id: 'collection.marketplace.discover.subtitle',
        })}
      />

      <div
        style={{
          maxWidth: '600px',
          marginInline: 'auto',
          paddingInline: 24,
          marginBottom: 24,
        }}
      >
        <Input
          size="large"
          placeholder={formatMessage({
            id: 'collection.marketplace.search.placeholder',
          })}
          allowClear
          onChange={(e) => handleSearchChange(e.currentTarget.value)}
          style={{ borderRadius: 24, }}
        />
      </div>

      {renderEmptyState() || (
        <>
          <Row gutter={[24, 24]}>
            {filteredCollections.map((collection) => (
              <Col
                key={collection.id}
                xs={24}
                sm={12}
                md={12}
                lg={8}
                xl={6}
                xxl={6}
              >
                <CollectionMarketplaceCard
                  collection={collection}
                  onClick={handleCardClick}
                  onSubscribe={handleSubscribe}
                  onUnsubscribe={handleUnsubscribe}
                  loading={subscribing === collection.id}
                />
              </Col>
            ))}
          </Row>

          {marketplacePagination.total > marketplacePagination.pageSize && (
            <div
              style={{
                textAlign: 'center',
                marginTop: 40,
                padding: '24px 0',
              }}
            >
              <Pagination
                current={marketplacePagination.current}
                pageSize={marketplacePagination.pageSize}
                total={marketplacePagination.total}
                onChange={handlePageChange}
                showSizeChanger
                showQuickJumper
                showTotal={(total, range) =>
                  formatMessage(
                    { id: 'common.pagination.total' },
                    { start: range[0], end: range[1], total },
                  )
                }
              />
            </div>
          )}
        </>
      )}
    </PageContainer>
  );
};

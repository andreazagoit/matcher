export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type MakeEmpty<T extends { [key: string]: unknown }, K extends keyof T> = { [_ in K]?: never };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Boolean: { input: boolean; output: boolean; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
  DateTime: { input: unknown; output: unknown; }
  JSON: { input: unknown; output: unknown; }
};

export enum ActivityLevel {
  Active = 'active',
  Light = 'light',
  Moderate = 'moderate',
  Sedentary = 'sedentary',
  VeryActive = 'very_active'
}

export type AddProfileItemInput = {
  content: Scalars['String']['input'];
  displayOrder?: InputMaybe<Scalars['Int']['input']>;
  promptKey?: InputMaybe<Scalars['String']['input']>;
  type: ProfileItemType;
};

export enum AttendeeStatus {
  Attended = 'attended',
  Going = 'going',
  Interested = 'interested'
}

export type Conversation = {
  __typename: 'Conversation';
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  initiator: User;
  lastMessage: Maybe<Message>;
  lastMessageAt: Maybe<Scalars['DateTime']['output']>;
  otherUser: User;
  recipient: User;
  source: Maybe<Scalars['String']['output']>;
  status: ConversationStatus;
  unreadCount: Maybe<Scalars['Int']['output']>;
  updatedAt: Scalars['DateTime']['output'];
};

export enum ConversationStatus {
  Active = 'active',
  Declined = 'declined',
  Request = 'request'
}

export type CreateEventInput = {
  currency?: InputMaybe<Scalars['String']['input']>;
  description?: InputMaybe<Scalars['String']['input']>;
  endsAt?: InputMaybe<Scalars['String']['input']>;
  lat?: InputMaybe<Scalars['Float']['input']>;
  location?: InputMaybe<Scalars['String']['input']>;
  lon?: InputMaybe<Scalars['Float']['input']>;
  maxAttendees?: InputMaybe<Scalars['Int']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
  spaceId: Scalars['ID']['input'];
  startsAt: Scalars['String']['input'];
  tags?: InputMaybe<Array<Scalars['String']['input']>>;
  title: Scalars['String']['input'];
};

export type CreateSpaceInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  joinPolicy?: InputMaybe<Scalars['String']['input']>;
  name: Scalars['String']['input'];
  slug?: InputMaybe<Scalars['String']['input']>;
  tags?: InputMaybe<Array<Scalars['String']['input']>>;
  visibility?: InputMaybe<Scalars['String']['input']>;
};

export type CreateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval: Scalars['String']['input'];
  name: Scalars['String']['input'];
  price: Scalars['Int']['input'];
};

export type CreateUserInput = {
  birthdate: Scalars['String']['input'];
  email: Scalars['String']['input'];
  gender?: InputMaybe<Gender>;
  name: Scalars['String']['input'];
};

export enum Drinking {
  Never = 'never',
  Regularly = 'regularly',
  Sometimes = 'sometimes'
}

export enum EducationLevel {
  Bachelor = 'bachelor',
  HighSchool = 'high_school',
  Master = 'master',
  MiddleSchool = 'middle_school',
  Other = 'other',
  Phd = 'phd',
  Vocational = 'vocational'
}

export enum Ethnicity {
  BlackAfrican = 'black_african',
  EastAsian = 'east_asian',
  HispanicLatino = 'hispanic_latino',
  Indigenous = 'indigenous',
  MiddleEastern = 'middle_eastern',
  Mixed = 'mixed',
  Other = 'other',
  PacificIslander = 'pacific_islander',
  SouthAsian = 'south_asian',
  WhiteCaucasian = 'white_caucasian'
}

export type Event = {
  __typename: 'Event';
  attendeeCount: Scalars['Int']['output'];
  attendees: Array<EventAttendee>;
  coordinates: Maybe<EventCoordinates>;
  createdAt: Scalars['DateTime']['output'];
  createdBy: Scalars['ID']['output'];
  /** ISO 4217 currency code, e.g. 'eur' */
  currency: Maybe<Scalars['String']['output']>;
  description: Maybe<Scalars['String']['output']>;
  endsAt: Maybe<Scalars['DateTime']['output']>;
  id: Scalars['ID']['output'];
  /** True when the event requires purchasing a ticket */
  isPaid: Scalars['Boolean']['output'];
  location: Maybe<Scalars['String']['output']>;
  maxAttendees: Maybe<Scalars['Int']['output']>;
  /** Status of the currently authenticated user for this event (null if not authenticated or not RSVP'd) */
  myAttendeeStatus: Maybe<AttendeeStatus>;
  /** Payment status for the currently authenticated user (null if free event or no purchase) */
  myPaymentStatus: Maybe<Scalars['String']['output']>;
  /** Ticket price in cents (null = free event) */
  price: Maybe<Scalars['Int']['output']>;
  /** The space this event belongs to */
  space: Maybe<Space>;
  spaceId: Scalars['ID']['output'];
  startsAt: Scalars['DateTime']['output'];
  status: EventStatus;
  tags: Array<Scalars['String']['output']>;
  title: Scalars['String']['output'];
  updatedAt: Scalars['DateTime']['output'];
};

export type EventAttendee = {
  __typename: 'EventAttendee';
  attendedAt: Maybe<Scalars['DateTime']['output']>;
  eventId: Scalars['ID']['output'];
  id: Scalars['ID']['output'];
  paymentStatus: Maybe<Scalars['String']['output']>;
  registeredAt: Scalars['DateTime']['output'];
  status: AttendeeStatus;
  user: Maybe<User>;
  userId: Scalars['ID']['output'];
};

export type EventCoordinates = {
  __typename: 'EventCoordinates';
  lat: Scalars['Float']['output'];
  lon: Scalars['Float']['output'];
};

export enum EventStatus {
  Cancelled = 'cancelled',
  Completed = 'completed',
  Draft = 'draft',
  Published = 'published'
}

export enum Gender {
  Man = 'man',
  NonBinary = 'non_binary',
  Woman = 'woman'
}

export enum HasChildren {
  No = 'no',
  Yes = 'yes'
}

export type Location = {
  __typename: 'Location';
  lat: Scalars['Float']['output'];
  lon: Scalars['Float']['output'];
};

export type Match = {
  __typename: 'Match';
  distanceKm: Maybe<Scalars['Float']['output']>;
  score: Scalars['Float']['output'];
  sharedEventIds: Array<Scalars['String']['output']>;
  sharedSpaceIds: Array<Scalars['String']['output']>;
  sharedTags: Array<Scalars['String']['output']>;
  user: MatchUser;
};

export type MatchUser = {
  __typename: 'MatchUser';
  birthdate: Maybe<Scalars['String']['output']>;
  gender: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  name: Scalars['String']['output'];
};

export type Member = {
  __typename: 'Member';
  currentPeriodEnd: Maybe<Scalars['DateTime']['output']>;
  id: Scalars['ID']['output'];
  joinedAt: Scalars['DateTime']['output'];
  role: Scalars['String']['output'];
  status: Scalars['String']['output'];
  subscriptionId: Maybe<Scalars['String']['output']>;
  tier: Maybe<MembershipTier>;
  user: User;
};

export type MembershipTier = {
  __typename: 'MembershipTier';
  currency: Scalars['String']['output'];
  description: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  interval: Scalars['String']['output'];
  isActive: Scalars['Boolean']['output'];
  name: Scalars['String']['output'];
  price: Scalars['Int']['output'];
  spaceId: Scalars['ID']['output'];
};

export type Message = {
  __typename: 'Message';
  content: Scalars['String']['output'];
  conversationId: Scalars['ID']['output'];
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  readAt: Maybe<Scalars['DateTime']['output']>;
  sender: User;
};

export type Mutation = {
  __typename: 'Mutation';
  addProfileItem: ProfileItem;
  approveMember: Member;
  archiveTier: Scalars['Boolean']['output'];
  /** Create a new event in a space. */
  createEvent: Event;
  createPost: Post;
  createSpace: Space;
  createTier: MembershipTier;
  createUser: User;
  deleteNotification: Scalars['Boolean']['output'];
  deletePost: Scalars['Boolean']['output'];
  deleteProfileItem: Scalars['Boolean']['output'];
  deleteSpace: Scalars['Boolean']['output'];
  deleteUser: Scalars['Boolean']['output'];
  joinSpace: Member;
  leaveSpace: Scalars['Boolean']['output'];
  markAllNotificationsRead: Scalars['Boolean']['output'];
  /** Mark all messages in a conversation as read. */
  markAsRead: Maybe<Scalars['Boolean']['output']>;
  /** Mark an event as completed. Attendees with status 'going' become 'attended'. */
  markEventCompleted: Event;
  markNotificationRead: Maybe<Notification>;
  removeMember: Scalars['Boolean']['output'];
  reorderProfileItems: Array<ProfileItem>;
  /** Respond to an event (going, interested). */
  respondToEvent: EventAttendee;
  /** Accept or decline a message request. */
  respondToRequest: Conversation;
  /** Send a message in an existing conversation. */
  sendMessage: Message;
  /** Send a message request to another user. Creates a conversation with status=request. */
  sendMessageRequest: Conversation;
  /** Update an existing event. */
  updateEvent: Event;
  updateLocation: User;
  updateMemberRole: Member;
  /** Set the user's declared interests (replaces previous declared tags). */
  updateMyInterests: Array<UserInterest>;
  updateProfileItem: ProfileItem;
  updateSpace: Space;
  updateTier: MembershipTier;
  updateUser: Maybe<User>;
};


export type MutationAddProfileItemArgs = {
  input: AddProfileItemInput;
};


export type MutationApproveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationArchiveTierArgs = {
  id: Scalars['ID']['input'];
};


export type MutationCreateEventArgs = {
  input: CreateEventInput;
};


export type MutationCreatePostArgs = {
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']>>;
  spaceId: Scalars['ID']['input'];
};


export type MutationCreateSpaceArgs = {
  input: CreateSpaceInput;
};


export type MutationCreateTierArgs = {
  input: CreateTierInput;
  spaceId: Scalars['ID']['input'];
};


export type MutationCreateUserArgs = {
  input: CreateUserInput;
};


export type MutationDeleteNotificationArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeletePostArgs = {
  postId: Scalars['ID']['input'];
};


export type MutationDeleteProfileItemArgs = {
  itemId: Scalars['ID']['input'];
};


export type MutationDeleteSpaceArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeleteUserArgs = {
  id: Scalars['ID']['input'];
};


export type MutationJoinSpaceArgs = {
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
};


export type MutationLeaveSpaceArgs = {
  spaceId: Scalars['ID']['input'];
};


export type MutationMarkAsReadArgs = {
  conversationId: Scalars['ID']['input'];
};


export type MutationMarkEventCompletedArgs = {
  eventId: Scalars['ID']['input'];
};


export type MutationMarkNotificationReadArgs = {
  id: Scalars['ID']['input'];
};


export type MutationRemoveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationReorderProfileItemsArgs = {
  itemIds: Array<Scalars['ID']['input']>;
};


export type MutationRespondToEventArgs = {
  eventId: Scalars['ID']['input'];
  status: AttendeeStatus;
};


export type MutationRespondToRequestArgs = {
  accept: Scalars['Boolean']['input'];
  conversationId: Scalars['ID']['input'];
};


export type MutationSendMessageArgs = {
  content: Scalars['String']['input'];
  conversationId: Scalars['ID']['input'];
};


export type MutationSendMessageRequestArgs = {
  content: Scalars['String']['input'];
  recipientId: Scalars['ID']['input'];
  source?: InputMaybe<Scalars['String']['input']>;
};


export type MutationUpdateEventArgs = {
  id: Scalars['ID']['input'];
  input: UpdateEventInput;
};


export type MutationUpdateLocationArgs = {
  lat: Scalars['Float']['input'];
  lon: Scalars['Float']['input'];
};


export type MutationUpdateMemberRoleArgs = {
  role: Scalars['String']['input'];
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationUpdateMyInterestsArgs = {
  tags: Array<Scalars['String']['input']>;
};


export type MutationUpdateProfileItemArgs = {
  input: UpdateProfileItemInput;
  itemId: Scalars['ID']['input'];
};


export type MutationUpdateSpaceArgs = {
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
};


export type MutationUpdateTierArgs = {
  id: Scalars['ID']['input'];
  input: UpdateTierInput;
};


export type MutationUpdateUserArgs = {
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
};

export type Notification = {
  __typename: 'Notification';
  createdAt: Scalars['DateTime']['output'];
  href: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  read: Scalars['Boolean']['output'];
  text: Scalars['String']['output'];
  type: Scalars['String']['output'];
};

export type Post = {
  __typename: 'Post';
  author: User;
  commentsCount: Maybe<Scalars['Int']['output']>;
  content: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  id: Scalars['ID']['output'];
  likesCount: Maybe<Scalars['Int']['output']>;
  mediaUrls: Maybe<Array<Scalars['String']['output']>>;
  space: Space;
};

/** A single item on a user's profile — either a photo or a prompt answer. */
export type ProfileItem = {
  __typename: 'ProfileItem';
  content: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  displayOrder: Scalars['Int']['output'];
  id: Scalars['ID']['output'];
  promptKey: Maybe<Scalars['String']['output']>;
  type: ProfileItemType;
  updatedAt: Scalars['DateTime']['output'];
  userId: Scalars['ID']['output'];
};

export enum ProfileItemType {
  Photo = 'photo',
  Prompt = 'prompt'
}

export type ProfileStatus = {
  __typename: 'ProfileStatus';
  hasProfile: Scalars['Boolean']['output'];
  updatedAt: Maybe<Scalars['DateTime']['output']>;
};

export type Query = {
  __typename: 'Query';
  /** Get all valid tags as a flat list. */
  allTags: Array<Scalars['String']['output']>;
  checkUsername: Scalars['Boolean']['output'];
  /** Get a single conversation by ID. */
  conversation: Maybe<Conversation>;
  /** Get active conversations for the authenticated user. */
  conversations: Array<Conversation>;
  /** Get a single event by ID. */
  event: Maybe<Event>;
  /** Get attendees for an event. */
  eventAttendees: Array<EventAttendee>;
  /**
   * Search upcoming events by tags.
   * matchAll=true requires ALL tags, false requires at least one.
   */
  eventsByTags: Array<Event>;
  /**
   * Find compatible matches for the authenticated user.
   * Uses tag overlap, shared spaces/events, proximity, and behavioral similarity.
   */
  findMatches: Array<Match>;
  me: Maybe<User>;
  /** Get pending message requests (inbox). */
  messageRequests: Array<Conversation>;
  /** Get messages for a conversation. */
  messages: Array<Message>;
  /** Get the authenticated user's interests with weights. */
  myInterests: Array<UserInterest>;
  mySpaces: Array<Space>;
  /** Get the authenticated user's upcoming events. */
  myUpcomingEvents: Array<Event>;
  notifications: Array<Notification>;
  profileItems: Array<ProfileItem>;
  /** Get the authenticated user's profile status. */
  profileStatus: ProfileStatus;
  /**
   * Get recommended events based on behavioral similarity and tag overlap.
   * Falls back to upcoming events for new users.
   */
  recommendedEvents: Array<Event>;
  /**
   * Get recommended spaces based on behavioral similarity and tag overlap.
   * Excludes spaces the user is already a member of.
   */
  recommendedSpaces: Array<Space>;
  space: Maybe<Space>;
  /** Get events for a specific space. */
  spaceEvents: Array<Event>;
  spaces: Array<Space>;
  /**
   * Search public spaces by tags.
   * matchAll=true requires ALL tags, false requires at least one.
   */
  spacesByTags: Array<Space>;
  /** Get tag categories (shared vocabulary for profiles, events, spaces). */
  tagCategories: Array<TagCategoryEntry>;
  unreadNotificationsCount: Scalars['Int']['output'];
  user: Maybe<User>;
  userFeed: Array<Post>;
  users: Array<User>;
};


export type QueryCheckUsernameArgs = {
  username: Scalars['String']['input'];
};


export type QueryConversationArgs = {
  id: Scalars['ID']['input'];
};


export type QueryEventArgs = {
  id: Scalars['ID']['input'];
};


export type QueryEventAttendeesArgs = {
  eventId: Scalars['ID']['input'];
};


export type QueryEventsByTagsArgs = {
  matchAll?: InputMaybe<Scalars['Boolean']['input']>;
  tags: Array<Scalars['String']['input']>;
};


export type QueryFindMatchesArgs = {
  gender?: InputMaybe<Array<Scalars['String']['input']>>;
  limit?: InputMaybe<Scalars['Int']['input']>;
  maxAge?: InputMaybe<Scalars['Int']['input']>;
  maxDistance?: Scalars['Float']['input'];
  minAge?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type QueryMessagesArgs = {
  conversationId: Scalars['ID']['input'];
};


export type QueryNotificationsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type QueryProfileItemsArgs = {
  userId: Scalars['ID']['input'];
};


export type QueryRecommendedEventsArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
};


export type QueryRecommendedSpacesArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
};


export type QuerySpaceArgs = {
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
};


export type QuerySpaceEventsArgs = {
  spaceId: Scalars['ID']['input'];
};


export type QuerySpacesByTagsArgs = {
  matchAll?: InputMaybe<Scalars['Boolean']['input']>;
  tags: Array<Scalars['String']['input']>;
};


export type QueryUserArgs = {
  username: Scalars['String']['input'];
};


export type QueryUserFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};

export enum RelationshipStyle {
  EthicalNonMonogamous = 'ethical_non_monogamous',
  Monogamous = 'monogamous',
  Open = 'open',
  Other = 'other'
}

export enum Religion {
  Buddhist = 'buddhist',
  Christian = 'christian',
  Hindu = 'hindu',
  Jewish = 'jewish',
  Muslim = 'muslim',
  None = 'none',
  Other = 'other',
  Spiritual = 'spiritual'
}

export enum Smoking {
  Never = 'never',
  Regularly = 'regularly',
  Sometimes = 'sometimes'
}

export type Space = {
  __typename: 'Space';
  createdAt: Scalars['DateTime']['output'];
  description: Maybe<Scalars['String']['output']>;
  feed: Maybe<Array<Post>>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  isActive: Maybe<Scalars['Boolean']['output']>;
  joinPolicy: Scalars['String']['output'];
  members: Maybe<Array<Member>>;
  membersCount: Maybe<Scalars['Int']['output']>;
  myMembership: Maybe<Member>;
  name: Scalars['String']['output'];
  slug: Scalars['String']['output'];
  stripeAccountEnabled: Scalars['Boolean']['output'];
  tags: Array<Scalars['String']['output']>;
  tiers: Maybe<Array<MembershipTier>>;
  type: Maybe<Scalars['String']['output']>;
  visibility: Scalars['String']['output'];
};


export type SpaceFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type SpaceMembersArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};

export type TagCategoryEntry = {
  __typename: 'TagCategoryEntry';
  category: Scalars['String']['output'];
  tags: Array<Scalars['String']['output']>;
};

export type UpdateEventInput = {
  currency?: InputMaybe<Scalars['String']['input']>;
  description?: InputMaybe<Scalars['String']['input']>;
  endsAt?: InputMaybe<Scalars['String']['input']>;
  lat?: InputMaybe<Scalars['Float']['input']>;
  location?: InputMaybe<Scalars['String']['input']>;
  lon?: InputMaybe<Scalars['Float']['input']>;
  maxAttendees?: InputMaybe<Scalars['Int']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
  startsAt?: InputMaybe<Scalars['String']['input']>;
  status?: InputMaybe<EventStatus>;
  tags?: InputMaybe<Array<Scalars['String']['input']>>;
  title?: InputMaybe<Scalars['String']['input']>;
};

export type UpdateProfileItemInput = {
  content?: InputMaybe<Scalars['String']['input']>;
  promptKey?: InputMaybe<Scalars['String']['input']>;
};

export type UpdateSpaceInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  image?: InputMaybe<Scalars['String']['input']>;
  joinPolicy?: InputMaybe<Scalars['String']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  tags?: InputMaybe<Array<Scalars['String']['input']>>;
  visibility?: InputMaybe<Scalars['String']['input']>;
};

export type UpdateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval?: InputMaybe<Scalars['String']['input']>;
  isActive?: InputMaybe<Scalars['Boolean']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
};

export type UpdateUserInput = {
  activityLevel?: InputMaybe<ActivityLevel>;
  birthdate?: InputMaybe<Scalars['String']['input']>;
  drinking?: InputMaybe<Drinking>;
  educationLevel?: InputMaybe<EducationLevel>;
  email?: InputMaybe<Scalars['String']['input']>;
  ethnicity?: InputMaybe<Ethnicity>;
  gender?: InputMaybe<Gender>;
  hasChildren?: InputMaybe<HasChildren>;
  heightCm?: InputMaybe<Scalars['Int']['input']>;
  jobTitle?: InputMaybe<Scalars['String']['input']>;
  languages?: InputMaybe<Array<Scalars['String']['input']>>;
  name?: InputMaybe<Scalars['String']['input']>;
  relationshipIntent?: InputMaybe<Array<Scalars['String']['input']>>;
  relationshipStyle?: InputMaybe<RelationshipStyle>;
  religion?: InputMaybe<Religion>;
  schoolName?: InputMaybe<Scalars['String']['input']>;
  sexualOrientation?: InputMaybe<Array<Scalars['String']['input']>>;
  smoking?: InputMaybe<Smoking>;
  username?: InputMaybe<Scalars['String']['input']>;
  wantsChildren?: InputMaybe<WantsChildren>;
};

/** User — demographic, auth, and location data */
export type User = {
  __typename: 'User';
  activityLevel: Maybe<ActivityLevel>;
  birthdate: Scalars['String']['output'];
  createdAt: Scalars['DateTime']['output'];
  drinking: Maybe<Drinking>;
  educationLevel: Maybe<EducationLevel>;
  email: Scalars['String']['output'];
  ethnicity: Maybe<Ethnicity>;
  gender: Maybe<Gender>;
  hasChildren: Maybe<HasChildren>;
  heightCm: Maybe<Scalars['Int']['output']>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  interests: Array<UserInterest>;
  jobTitle: Maybe<Scalars['String']['output']>;
  languages: Array<Scalars['String']['output']>;
  location: Maybe<Location>;
  locationUpdatedAt: Maybe<Scalars['DateTime']['output']>;
  name: Scalars['String']['output'];
  profileItems: Array<ProfileItem>;
  relationshipIntent: Array<Scalars['String']['output']>;
  relationshipStyle: Maybe<RelationshipStyle>;
  religion: Maybe<Religion>;
  schoolName: Maybe<Scalars['String']['output']>;
  sexualOrientation: Array<Scalars['String']['output']>;
  smoking: Maybe<Smoking>;
  updatedAt: Scalars['DateTime']['output'];
  username: Maybe<Scalars['String']['output']>;
  wantsChildren: Maybe<WantsChildren>;
};

export type UserInterest = {
  __typename: 'UserInterest';
  tag: Scalars['String']['output'];
  weight: Scalars['Float']['output'];
};

export enum WantsChildren {
  No = 'no',
  Open = 'open',
  Yes = 'yes'
}

export type ConversationFieldsFragment = { __typename: 'Conversation', id: string, status: ConversationStatus, source: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null };

export type MessageFieldsFragment = { __typename: 'Message', id: string, conversationId: string, content: string, readAt: unknown | null, createdAt: unknown, sender: { __typename: 'User', id: string, name: string, image: string | null } };

export type GetConversationsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetConversationsQuery = { conversations: Array<{ __typename: 'Conversation', id: string, status: ConversationStatus, source: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null }> };

export type GetMessageRequestsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMessageRequestsQuery = { messageRequests: Array<{ __typename: 'Conversation', id: string, status: ConversationStatus, source: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null }> };

export type GetRecentConversationsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetRecentConversationsQuery = { conversations: Array<{ __typename: 'Conversation', id: string, unreadCount: number | null, otherUser: { __typename: 'User', name: string } }> };

export type GetMessagesQueryVariables = Exact<{
  conversationId: Scalars['ID']['input'];
}>;


export type GetMessagesQuery = { messages: Array<{ __typename: 'Message', id: string, conversationId: string, content: string, readAt: unknown | null, createdAt: unknown, sender: { __typename: 'User', id: string, name: string, image: string | null } }>, conversation: { __typename: 'Conversation', id: string, status: ConversationStatus, otherUser: { __typename: 'User', id: string, name: string, image: string | null } } | null };

export type SendMessageRequestMutationVariables = Exact<{
  recipientId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
  source?: InputMaybe<Scalars['String']['input']>;
}>;


export type SendMessageRequestMutation = { sendMessageRequest: { __typename: 'Conversation', id: string, status: ConversationStatus, source: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null } };

export type RespondToRequestMutationVariables = Exact<{
  conversationId: Scalars['ID']['input'];
  accept: Scalars['Boolean']['input'];
}>;


export type RespondToRequestMutation = { respondToRequest: { __typename: 'Conversation', id: string, status: ConversationStatus, source: string | null, lastMessageAt: unknown | null, createdAt: unknown, updatedAt: unknown, unreadCount: number | null, otherUser: { __typename: 'User', id: string, name: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: unknown } | null } };

export type SendMessageMutationVariables = Exact<{
  conversationId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
}>;


export type SendMessageMutation = { sendMessage: { __typename: 'Message', id: string, content: string, createdAt: unknown } };

export type MarkAsReadMutationVariables = Exact<{
  conversationId: Scalars['ID']['input'];
}>;


export type MarkAsReadMutation = { markAsRead: boolean | null };

export type SpaceEventsQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type SpaceEventsQuery = { spaceEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, status: EventStatus, attendeeCount: number, createdBy: string, tags: Array<string>, price: number | null, currency: string | null, isPaid: boolean }> };

export type GetEventQueryVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type GetEventQuery = { event: { __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, status: EventStatus, attendeeCount: number, myAttendeeStatus: AttendeeStatus | null, myPaymentStatus: string | null, tags: Array<string>, spaceId: string, createdBy: string, createdAt: unknown, price: number | null, currency: string | null, isPaid: boolean, coordinates: { __typename: 'EventCoordinates', lat: number, lon: number } | null, space: { __typename: 'Space', id: string, name: string, slug: string, visibility: string, stripeAccountEnabled: boolean, myMembership: { __typename: 'Member', role: string } | null } | null, attendees: Array<{ __typename: 'EventAttendee', id: string, userId: string, status: AttendeeStatus, registeredAt: unknown, paymentStatus: string | null, user: { __typename: 'User', id: string, name: string, username: string | null } | null }> } | null };

export type UpdateEventMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateEventInput;
}>;


export type UpdateEventMutation = { updateEvent: { __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, status: EventStatus, tags: Array<string>, price: number | null, currency: string | null } };

export type MyUpcomingEventsQueryVariables = Exact<{ [key: string]: never; }>;


export type MyUpcomingEventsQuery = { myUpcomingEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, status: EventStatus, attendeeCount: number, tags: Array<string>, spaceId: string, coordinates: { __typename: 'EventCoordinates', lat: number, lon: number } | null }> };

export type RecommendedEventsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type RecommendedEventsQuery = { recommendedEvents: Array<{ __typename: 'Event', id: string, title: string, description: string | null, location: string | null, startsAt: unknown, endsAt: unknown | null, maxAttendees: number | null, status: EventStatus, attendeeCount: number, tags: Array<string>, spaceId: string, coordinates: { __typename: 'EventCoordinates', lat: number, lon: number } | null }> };

export type CreateEventMutationVariables = Exact<{
  input: CreateEventInput;
}>;


export type CreateEventMutation = { createEvent: { __typename: 'Event', id: string } };

export type RespondToEventMutationVariables = Exact<{
  eventId: Scalars['ID']['input'];
  status: AttendeeStatus;
}>;


export type RespondToEventMutation = { respondToEvent: { __typename: 'EventAttendee', id: string, status: AttendeeStatus } };

export type MarkEventCompletedMutationVariables = Exact<{
  eventId: Scalars['ID']['input'];
}>;


export type MarkEventCompletedMutation = { markEventCompleted: { __typename: 'Event', id: string, status: EventStatus } };

export type UpdateMyInterestsMutationVariables = Exact<{
  tags: Array<Scalars['String']['input']> | Scalars['String']['input'];
}>;


export type UpdateMyInterestsMutation = { updateMyInterests: Array<{ __typename: 'UserInterest', tag: string, weight: number }> };

export type GetFindMatchesQueryVariables = Exact<{
  maxDistance?: Scalars['Float']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
  gender?: InputMaybe<Array<Scalars['String']['input']> | Scalars['String']['input']>;
  minAge?: InputMaybe<Scalars['Int']['input']>;
  maxAge?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetFindMatchesQuery = { findMatches: Array<{ __typename: 'Match', score: number, distanceKm: number | null, sharedTags: Array<string>, sharedSpaceIds: Array<string>, sharedEventIds: Array<string>, user: { __typename: 'MatchUser', id: string, name: string, image: string | null, gender: string | null, birthdate: string | null } }> };

export type GetProfileStatusQueryVariables = Exact<{ [key: string]: never; }>;


export type GetProfileStatusQuery = { profileStatus: { __typename: 'ProfileStatus', hasProfile: boolean, updatedAt: unknown | null } };

export type UpdateMemberRoleMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
  role: Scalars['String']['input'];
}>;


export type UpdateMemberRoleMutation = { updateMemberRole: { __typename: 'Member', id: string, role: string } };

export type RemoveMemberMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
}>;


export type RemoveMemberMutation = { removeMember: boolean };

export type NotificationFieldsFragment = { __typename: 'Notification', id: string, type: string, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown };

export type GetNotificationsQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetNotificationsQuery = { notifications: Array<{ __typename: 'Notification', id: string, type: string, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown }> };

export type GetUnreadNotificationsCountQueryVariables = Exact<{ [key: string]: never; }>;


export type GetUnreadNotificationsCountQuery = { unreadNotificationsCount: number };

export type MarkNotificationReadMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type MarkNotificationReadMutation = { markNotificationRead: { __typename: 'Notification', id: string, type: string, text: string, image: string | null, href: string | null, read: boolean, createdAt: unknown } | null };

export type MarkAllNotificationsReadMutationVariables = Exact<{ [key: string]: never; }>;


export type MarkAllNotificationsReadMutation = { markAllNotificationsRead: boolean };

export type DeleteNotificationMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type DeleteNotificationMutation = { deleteNotification: boolean };

export type PostFieldsFragment = { __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } };

export type GetUserFeedQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetUserFeedQuery = { userFeed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } }> };

export type GetSpaceFeedQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceFeedQuery = { space: { __typename: 'Space', feed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: unknown, author: { __typename: 'User', id: string, name: string, image: string | null }, space: { __typename: 'Space', id: string, name: string, slug: string } }> | null } | null };

export type CreatePostMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']> | Scalars['String']['input']>;
}>;


export type CreatePostMutation = { createPost: { __typename: 'Post', id: string } };

export type DeletePostMutationVariables = Exact<{
  postId: Scalars['ID']['input'];
}>;


export type DeletePostMutation = { deletePost: boolean };

export type ProfileItemFieldsFragment = { __typename: 'ProfileItem', id: string, userId: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown };

export type GetProfileItemsQueryVariables = Exact<{
  userId: Scalars['ID']['input'];
}>;


export type GetProfileItemsQuery = { profileItems: Array<{ __typename: 'ProfileItem', id: string, userId: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown }> };

export type AddProfileItemMutationVariables = Exact<{
  input: AddProfileItemInput;
}>;


export type AddProfileItemMutation = { addProfileItem: { __typename: 'ProfileItem', id: string, userId: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown } };

export type UpdateProfileItemMutationVariables = Exact<{
  itemId: Scalars['ID']['input'];
  input: UpdateProfileItemInput;
}>;


export type UpdateProfileItemMutation = { updateProfileItem: { __typename: 'ProfileItem', id: string, userId: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown } };

export type DeleteProfileItemMutationVariables = Exact<{
  itemId: Scalars['ID']['input'];
}>;


export type DeleteProfileItemMutation = { deleteProfileItem: boolean };

export type ReorderProfileItemsMutationVariables = Exact<{
  itemIds: Array<Scalars['ID']['input']> | Scalars['ID']['input'];
}>;


export type ReorderProfileItemsMutation = { reorderProfileItems: Array<{ __typename: 'ProfileItem', id: string, userId: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number, createdAt: unknown, updatedAt: unknown }> };

export type SpaceFieldsFragment = { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean };

export type GetAllSpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetAllSpacesQuery = { spaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean }> };

export type GetRecommendedSpacesQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetRecommendedSpacesQuery = { recommendedSpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean }> };

export type GetMySpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMySpacesQuery = { mySpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean, myMembership: { __typename: 'Member', role: string } | null }> };

export type GetSpaceQueryVariables = Exact<{
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
  membersLimit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceQuery = { space: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean, myMembership: { __typename: 'Member', id: string, role: string, status: string, tier: { __typename: 'MembershipTier', id: string, name: string, price: number, interval: string } | null } | null, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string }> | null, members: Array<{ __typename: 'Member', id: string, role: string, status: string, joinedAt: unknown, tier: { __typename: 'MembershipTier', name: string } | null, user: { __typename: 'User', id: string, name: string, email: string } }> | null } | null };

export type CreateSpaceMutationVariables = Exact<{
  input: CreateSpaceInput;
}>;


export type CreateSpaceMutation = { createSpace: { __typename: 'Space', id: string, name: string, slug: string } };

export type UpdateSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
}>;


export type UpdateSpaceMutation = { updateSpace: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, tags: Array<string>, visibility: string, joinPolicy: string, createdAt: unknown, isActive: boolean | null, membersCount: number | null, type: string | null, stripeAccountEnabled: boolean } };

export type DeleteSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type DeleteSpaceMutation = { deleteSpace: boolean };

export type JoinSpaceMutationVariables = Exact<{
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
}>;


export type JoinSpaceMutation = { joinSpace: { __typename: 'Member', id: string, status: string } };

export type LeaveSpaceMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type LeaveSpaceMutation = { leaveSpace: boolean };

export type TierFieldsFragment = { __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string };

export type GetSpaceTiersQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type GetSpaceTiersQuery = { space: { __typename: 'Space', id: string, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string }> | null } | null };

export type CreateTierMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  input: CreateTierInput;
}>;


export type CreateTierMutation = { createTier: { __typename: 'MembershipTier', id: string } };

export type ArchiveTierMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type ArchiveTierMutation = { archiveTier: boolean };

export type UserFieldsFragment = { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null };

export type GetMeQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMeQuery = { me: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null } | null };

export type GetUserQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type GetUserQuery = { user: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, interests: Array<{ __typename: 'UserInterest', tag: string, weight: number }>, profileItems: Array<{ __typename: 'ProfileItem', id: string, type: ProfileItemType, promptKey: string | null, content: string, displayOrder: number }> } | null };

export type GetUserWithCardsQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type GetUserWithCardsQuery = { user: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, interests: Array<{ __typename: 'UserInterest', tag: string, weight: number }> } | null };

export type CheckUsernameQueryVariables = Exact<{
  username: Scalars['String']['input'];
}>;


export type CheckUsernameQuery = { checkUsername: boolean };

export type UpdateUserMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
}>;


export type UpdateUserMutation = { updateUser: { __typename: 'User', id: string, username: string | null, name: string, email: string, birthdate: string, gender: Gender | null, image: string | null, createdAt: unknown, updatedAt: unknown, sexualOrientation: Array<string>, heightCm: number | null, relationshipIntent: Array<string>, relationshipStyle: RelationshipStyle | null, hasChildren: HasChildren | null, wantsChildren: WantsChildren | null, religion: Religion | null, smoking: Smoking | null, drinking: Drinking | null, activityLevel: ActivityLevel | null, jobTitle: string | null, educationLevel: EducationLevel | null, schoolName: string | null, languages: Array<string>, ethnicity: Ethnicity | null, interests: Array<{ __typename: 'UserInterest', tag: string, weight: number }> } | null };
